#!/bin/bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ALLOC
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
#export NCCL_IB_DISABLE=1

# nccl default channels is 32 to strip transfers across many in parallel, too much noise
export NCCL_MAX_P2P_NCHANNELS=1

# force GDR even if GPU is far away from device
export NCCL_NET_GDR_LEVEL=SYS

# Stream output: show non-NCCL lines on stdout, save NCCL lines to tempfile for proxy filtering
NCCL_LOG=$(mktemp)
if [[ "$*" == *--verbose* ]]; then
  # Show all output; still save NCCL lines for proxy analysis
  AWK_FILTER='/NCCL INFO|NCCL WARN|NCCL TRACE/ { print >> logfile } { print }'
else
  AWK_FILTER='/NCCL INFO|NCCL WARN|NCCL TRACE/ { print >> logfile; next } { print }'
fi

if [[ "$*" == *--spinlock* ]]; then
  FIFO=$(mktemp -u)
  mkfifo "$FIFO"

  mpirun --tag-output -np 2 ./single_out.sh "$@" >"$FIFO" 2>&1 &
  MPIRUN_PID=$!

  awk -v logfile="$NCCL_LOG" "$AWK_FILTER" <"$FIFO" &

  read -r -p $'\nPress Enter to release spinlock (sends SIGUSR1)...\n' </dev/tty

  kill -USR1 $(pgrep -u "$(whoami)" ring_pattern_mp) 2>/dev/null

  wait "$MPIRUN_PID"
  rm -f "$FIFO"
else
  mpirun --tag-output -np 2 ./single_out.sh "$@" 2>&1 | \
    awk -v logfile="$NCCL_LOG" "$AWK_FILTER"
fi

echo ""
# Extract TIDs for each proxy thread type
PROGRESS_TIDS=$(grep -oP '\d+:\d+:\K\d+(?= \[\d\] NCCL INFO \[Proxy Progress\])' "$NCCL_LOG" | sort -u | paste -sd'|')
SERVICE_TIDS=$(grep -oP '\d+:\d+:\K\d+(?= \[\d\] NCCL INFO \[Proxy Service\] )' "$NCCL_LOG" | sort -u | paste -sd'|')
UDS_TIDS=$(grep -oP '\d+:\d+:\K\d+(?= \[\d\] NCCL INFO \[Proxy Service UDS\])' "$NCCL_LOG" | sort -u | paste -sd'|')
ALL_PROXY_TIDS=$(echo "$PROGRESS_TIDS|$SERVICE_TIDS|$UDS_TIDS" | sed 's/^|//;s/|$//' | sed 's/||/|/g')

# ANSI colors: Service=cyan, UDS=yellow, Progress=green
C_SVC=$'\033[36m'   # cyan
C_UDS=$'\033[33m'   # yellow
C_PRG=$'\033[32m'   # green
C_RST=$'\033[0m'

echo "=== Proxy thread TIDs: ${C_SVC}Service=$SERVICE_TIDS${C_RST} ${C_UDS}UDS=$UDS_TIDS${C_RST} ${C_PRG}Progress=$PROGRESS_TIDS${C_RST} ==="
echo ""
echo "=== Proxy thread output only ==="
if [ -n "$ALL_PROXY_TIDS" ]; then
  grep -P ":\d+:($ALL_PROXY_TIDS) " "$NCCL_LOG" | \
    awk -v svc="$SERVICE_TIDS" -v uds="$UDS_TIDS" -v prg="$PROGRESS_TIDS" \
        -v c_svc="$C_SVC" -v c_uds="$C_UDS" -v c_prg="$C_PRG" -v c_rst="$C_RST" '
    {
      # extract TID from hostname:PID:TID
      match($0, /:[0-9]+:([0-9]+) /, m)
      tid = m[1]
      if (tid != "" && index("|" prg "|", "|" tid "|")) { printf "%s%s%s\n", c_prg, $0, c_rst }
      else if (tid != "" && index("|" uds "|", "|" tid "|")) { printf "%s%s%s\n", c_uds, $0, c_rst }
      else if (tid != "" && index("|" svc "|", "|" tid "|")) { printf "%s%s%s\n", c_svc, $0, c_rst }
      else print
    }'
else
  echo "(no proxy threads found)"
fi
