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
#include <signal.h>
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

static volatile sig_atomic_t g_resume = 0;
static bool g_spinlock_enabled = false;

static void usr1_handler(int) { g_resume = 1; }

const size_t count = 256 * 1024 * 1024; // 256M floats = 1GB
const size_t size_bytes = count * sizeof(float);

void setupComm(int my_rank, int total_ranks, int local_device,
               ncclComm_t *comm, int *next, int *prev);

void fill_buffs(int my_rank, 
              float **h_sendbuff, float **d_sendbuff, 
              float **h_recvbuff, float **d_recvbuff);

void allocateBuffers(int my_rank, float **h_sendbuff,
                     float **h_recvbuff, float **d_sendbuff, float **d_recvbuff,
                     cudaStream_t *stream);

void verifyResults(int my_rank, int prev, float *h_recvbuff, float *d_recvbuff,
                   size_t count);

void cleanup(int my_rank, float *h_sendbuff, float *h_recvbuff,
             float *d_sendbuff, float *d_recvbuff, cudaStream_t stream,
             ncclComm_t comm);

void communicate(int &my_rank, ncclComm_t &comm, int &next, int &prev,
               const size_t &count, float *&d_sendbuff, float *&d_recvbuff,
               cudaStream_t &stream);

void show_mem(int my_rank, ncclComm_t comm);

void *ringPattern(int my_rank, int total_ranks, int local_device,
                  int devices_per_rank) {

  ncclComm_t comm;
  int next, prev;
  setupComm(my_rank, total_ranks, local_device, &comm, &next, &prev);


  float *h_sendbuff, *h_recvbuff, *d_sendbuff, *d_recvbuff;
  cudaStream_t stream;
  allocateBuffers(my_rank, &h_sendbuff, &h_recvbuff, &d_sendbuff,
                  &d_recvbuff, &stream);

  // ========================================================================
  // STEP 4: Execute Ring Communication
  // ========================================================================

  communicate(my_rank, comm, next, prev, count, d_sendbuff, d_recvbuff, stream);
  
  verifyResults(my_rank, prev, h_recvbuff, d_recvbuff, count);
  for (size_t j = 0; j < count; j++) {
    h_recvbuff[j] = -1.0f; // Initialize recv buffer to an invalid value 
  }

  // show_mem(my_rank, comm);
  ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
  if (g_spinlock_enabled) {
    if (my_rank == 0)
      printf("Spinning — send SIGUSR1 (kill -USR1 %d) to continue\n", getpid());
    while (!g_resume) {}
  }
  ncclCommResume(comm);


  verifyResults(my_rank, prev, h_recvbuff, d_recvbuff, count);

  fill_buffs(my_rank, &h_sendbuff, &d_sendbuff, &h_recvbuff, &d_recvbuff);

  communicate(my_rank, comm, next, prev, count, d_sendbuff, d_recvbuff, stream);
  verifyResults(my_rank, prev, h_recvbuff, d_recvbuff, count);

  // ========================================================================
  // STEP 6: Cleanup
  // ========================================================================
  cleanup(my_rank, h_sendbuff, h_recvbuff, d_sendbuff, d_recvbuff, stream,
          comm);
  // pthread_exit(nullptr);
  return NULL;
}

int main(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--spinlock") == 0) {
      g_spinlock_enabled = true;
      signal(SIGUSR1, usr1_handler);
    }
  }
  return run_example(argc, argv, ringPattern);
}


void show_mem(int my_rank, ncclComm_t comm) {
  double GB_TO_B = 1024.0 * 1024 * 1024;
  size_t suspendMem = 0;
  ncclCommMemStats(comm, ncclStatGpuMemSuspend, &suspendMem);
  size_t persistMem = 0;
  ncclCommMemStats(comm, ncclStatGpuMemPersist, &persistMem);
  size_t totalMem = 0;
  ncclCommMemStats(comm, ncclStatGpuMemTotal, &totalMem);
  printf("suspended %.2lf GB\npersisted %.2lf\n total = %.2lf\n", suspendMem / GB_TO_B, persistMem / GB_TO_B, totalMem / GB_TO_B);
  // printf("T");
}


void setupComm(int my_rank, int total_ranks, int local_device,
               ncclComm_t *comm, int *next, int *prev) {
  ncclUniqueId nccl_unique_id;
  if (my_rank == 0) {
    printf("Starting NCCL ring communication example with %d ranks\n",
           total_ranks);
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  util_broadcast(0, my_rank, &nccl_unique_id);

  CUDACHECK(cudaSetDevice(local_device));

  NCCLCHECK(ncclCommInitRank(comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d communicator initialized on device %d\n", my_rank,
         local_device);

  *next = (my_rank + 1) % total_ranks;
  *prev = (my_rank - 1 + total_ranks) % total_ranks;

  if (my_rank == 0) {
    printf("Ring topology: rank 0 -> ... -> rank %d -> rank 0\n",
           total_ranks - 1);
  }
  printf("  Rank %d sends to rank %d, receives from rank %d\n", my_rank, *next,
         *prev);
}

void fill_buffs(int my_rank,
              float **h_sendbuff, float **d_sendbuff, 
              float **h_recvbuff, float **d_recvbuff) {
  for (size_t j = 0; j < count; j++) {
    (*h_sendbuff)[j] = (float)(my_rank * 1000 + j % 1000);
    (*h_recvbuff)[j] = -1.0f; // Initialize recv buffer to an invalid value 
  }
  CUDACHECK(
      cudaMemcpy(*d_sendbuff, *h_sendbuff, size_bytes, cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(*d_recvbuff, *h_recvbuff, size_bytes, cudaMemcpyHostToDevice));

}

void allocateBuffers(int my_rank, float **h_sendbuff,
                     float **h_recvbuff, float **d_sendbuff, float **d_recvbuff,
                     cudaStream_t *stream) {

  if (my_rank == 0) {
    printf("Ring transfer with %zu elements (%.2f GB per rank)\n", count,
           size_bytes / (1024.0 * 1024.0 * 1024.0));
  }

  *h_sendbuff = (float *)malloc(size_bytes);
  *h_recvbuff = (float *)malloc(size_bytes);
  CUDACHECK(cudaMalloc((void **)d_sendbuff, size_bytes));
  CUDACHECK(cudaMalloc((void **)d_recvbuff, size_bytes));

  fill_buffs(my_rank, h_sendbuff, d_sendbuff, h_recvbuff, d_recvbuff);

  CUDACHECK(cudaStreamCreate(stream));
}

void verifyResults(int my_rank, int prev, float *h_recvbuff, float *d_recvbuff,
                   size_t count) {
  CUDACHECK(
      cudaMemcpy(h_recvbuff, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));

  float expected = (float)(prev * 1000);
  bool correct = (h_recvbuff[0] == expected);

  printf("  Rank %d received data from rank %d: %s\n", my_rank, prev,
         correct ? "CORRECT" : "ERROR");
  if (!correct) {
    printf("    Expected %.0f, got %.0f\n", expected, h_recvbuff[0]);
  }
}

void cleanup(int my_rank, float *h_sendbuff, float *h_recvbuff,
             float *d_sendbuff, float *d_recvbuff, cudaStream_t stream,
             ncclComm_t comm) {
  printf("Freeing resources for rank %d\n", my_rank);
  free(h_sendbuff);
  free(h_recvbuff);
  CUDACHECK(cudaFree(d_sendbuff));
  CUDACHECK(cudaFree(d_recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  printf("  Rankinger %d communicator finalized\n", my_rank);
}

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
void communicate(int &my_rank, ncclComm_t &comm, int &next, int &prev,
               const size_t &count, float *&d_sendbuff, float *&d_recvbuff,
               cudaStream_t &stream) {
  if (my_rank == 0) {
    printf("Executing ring communication\n");
  }
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(d_sendbuff, count, ncclFloat, next, comm, stream));
  NCCLCHECK(ncclRecv(d_recvbuff, count, ncclFloat, prev, comm, stream));
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaStreamSynchronize(stream));
}