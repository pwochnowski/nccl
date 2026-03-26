/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "utils.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * NCCL Ring Pattern Example - Multiple Process Version
 *
 * This example demonstrates the ring communication pattern using one
 * process (or pthread) per GPU via the utils.cc framework. Compare with
 * 01_ring_pattern which uses a single process managing all GPUs.
 *
 * Learning Objectives:
 * - Understand ring topology with one device per process/thread
 * - Learn ncclCommInitRank-based communicator setup
 * - See why ncclGroupStart/End are still needed when multiple
 *   communicators share the same process (pthread mode)
 * - Practice using the utils.cc framework for multi-process examples
 *
 */

/*
 * This function is called once per rank (MPI process or pthread).
 * Each rank manages exactly one GPU.
 */
void *ringPattern(int my_rank, int total_ranks, int local_device,
                  int devices_per_rank) {

  // ========================================================================
  // STEP 1: Initialize NCCL Communicator
  // ========================================================================

  ncclUniqueId nccl_unique_id;
  if (my_rank == 0) {
    printf("Starting NCCL ring communication example with %d ranks\n",
           total_ranks);
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID to all ranks
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  CUDACHECK(cudaSetDevice(local_device));

  // Initialize communicator using ncclCommInitRank (one rank per device)
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d communicator initialized on device %d\n", my_rank,
         local_device);

  // ========================================================================
  // STEP 2: Compute Ring Neighbors
  // ========================================================================

  int next = (my_rank + 1) % total_ranks;
  int prev = (my_rank - 1 + total_ranks) % total_ranks;

  if (my_rank == 0) {
    printf("Ring topology: rank 0 -> ... -> rank %d -> rank 0\n",
           total_ranks - 1);
  }
  printf("  Rank %d sends to rank %d, receives from rank %d\n", my_rank, next,
        prev);

  // ========================================================================
  // STEP 3: Allocate and Initialize Buffers
  // ========================================================================

  const size_t count = 256 * 1024 * 1024; // 256M floats = 1GB
  const size_t size_bytes = count * sizeof(float);

  if (my_rank == 0) {
    printf("Ring transfer with %zu elements (%.2f GB per rank)\n", count,
           size_bytes / (1024.0 * 1024.0 * 1024.0));
  }

  float *h_sendbuff = (float *)malloc(size_bytes);
  float *h_recvbuff = (float *)malloc(size_bytes);
  float *d_sendbuff;
  float *d_recvbuff;
  CUDACHECK(cudaMalloc((void **)&d_sendbuff, size_bytes));
  CUDACHECK(cudaMalloc((void **)&d_recvbuff, size_bytes));

  // Initialize data with rank-specific pattern for verification
  for (size_t j = 0; j < count; j++) {
    h_sendbuff[j] = (float)(my_rank * 1000 + j % 1000);
  }
  CUDACHECK(
      cudaMemcpy(d_sendbuff, h_sendbuff, size_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // ========================================================================
  // STEP 4: Execute Ring Communication
  // ========================================================================

  /*
   * ncclGroupStart/End are required whenever multiple communicators exist
   * in the same process — including the pthread case. NCCL uses a
   * process-wide group counter, so each thread's ncclGroupStart/End
   * pairs nest correctly: the operations are batched until the last
   * thread calls ncclGroupEnd, then all are flushed together.
   *
   * With MPI (separate processes), group calls would not be strictly
   * necessary since each process has only one communicator, but they
   * are harmless and keep the code portable across both backends.
   */
  if (my_rank == 0) {
    printf("Executing ring communication\n");

  }
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(d_sendbuff, count, ncclFloat, next, comm, stream));
  NCCLCHECK(ncclRecv(d_recvbuff, count, ncclFloat, prev, comm, stream));
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaStreamSynchronize(stream));

  if (my_rank == 0) {
    printf("Ring communication completed successfully\n");
  }

  // if (my_rank == 0) {
  // ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
  // ncclCommResume(comm);
  // }
  // ========================================================================
  // STEP 5: Verify Data Correctness
  // ========================================================================

  CUDACHECK(
      cudaMemcpy(h_recvbuff, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));

  // Each rank should have received data from its predecessor
  float expected = (float)(prev * 1000);
  bool correct = (h_recvbuff[0] == expected);

    printf("  Rank %d received data from rank %d: %s\n", my_rank, prev,
          correct ? "CORRECT" : "ERROR");
    if (!correct) {
      printf("    Expected %.0f, got %.0f\n", expected, h_recvbuff[0]);
    }

  // ========================================================================
  // STEP 6: Cleanup
  // ========================================================================
  printf("Freeing resources for rank %d\n", my_rank);
  free(h_sendbuff);
  free(h_recvbuff);
  CUDACHECK(cudaFree(d_sendbuff));
  CUDACHECK(cudaFree(d_recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  printf("  Rankinger %d communicator finalized\n", my_rank);
  // pthread_exit(nullptr);
  return NULL;
}

int main(int argc, char *argv[]) {
  return run_example(argc, argv, ringPattern);
}
