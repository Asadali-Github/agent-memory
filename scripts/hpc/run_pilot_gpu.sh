#!/bin/bash
# FILE: run_pilot_gpu.sh — CERES pilot run. THIS VERSION WORKS (job 2847608,
# 19 Jul 2026: all 5 conditions x 3 items completed on gpu-1-1 RTX2080).
# Submit from ~/agent-memory:  qsub scripts/hpc/run_pilot_gpu.sh
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N agentmem-pilot
#$ -q gpu.q@gpu-1-1,gpu.q@gpu-1-4
#$ -l gpu=1
source /usr/local/gpuallocation.sh

# --- pick a python3.8 with working lzma+ssl (stdlib varies per node) ---
PY=""
for CAND in /usr/bin/python38 /usr/bin/python3.8 /usr/local/bin/python3.8; do
  [ -x "$CAND" ] && "$CAND" -c "import lzma, ssl" 2>/dev/null && PY=$CAND && break
done
[ -z "$PY" ] && { echo "NO usable python3.8 on $(hostname)"; exit 1; }
echo "Using python: $PY"

export PATH="$HOME/.local/bin:$PATH"
CUPTI=$(ls -d /usr/local/cuda-12*/extras/CUPTI/lib64 2>/dev/null | tail -1)
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$CUPTI:$LD_LIBRARY_PATH"

# --- shims for the GPU nodes' incomplete python builds ---
# sqlite3: real replacement via pysqlite3-binary + PYTHONPATH shadow
$PY -m pip install --user --quiet pysqlite3-binary
mkdir -p ~/pyshim/pyairports
printf 'from pysqlite3.dbapi2 import *\nfrom pysqlite3 import dbapi2\n' > ~/pyshim/sqlite3.py
# pyairports: empty stand-in (outlines imports it; never used by our requests)
printf 'AIRPORT_LIST = []\n' > ~/pyshim/pyairports/airports.py
touch ~/pyshim/pyairports/__init__.py
export PYTHONPATH="$HOME/pyshim:$PYTHONPATH"
$PY -c "import sqlite3; print('sqlite3 shim OK', sqlite3.sqlite_version)" || exit 1

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=$((10000 + RANDOM % 20000))
export HPC_BASE_URL="http://localhost:$PORT/v1"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Port: $PORT"

$PY -m pip install --user --quiet "urllib3<2"   # urllib3 v2 breaks on CentOS7 OpenSSL
$PY -c "import torch; print('torch OK', torch.__version__, 'cuda:', torch.cuda.is_available())" || exit 1

# RTX2080 (8GB): fp16 forced (no bf16 on CC 7.5), 2048 ctx + max 4 seqs fits
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  > vllm_server_$JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED — see vllm_server_$JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM is up."

$PY scripts/run_experiment.py --task custom_ds --all-conditions --seed 1 --limit 3

kill $VLLM_PID
echo "PILOT DONE — inspect results/runs/ and this log."
