#!/bin/bash
# FILE: run_grid_gpu.sh — CERES FULL experiment grid. Same proven recipe as
# run_pilot_gpu.sh (job 2847608). Submit AFTER inspecting pilot results and
# freezing configs:  qsub scripts/hpc/run_grid_gpu.sh
# Resumable: if it dies or hits queue limits, qsub again — done cells skip.
# NOTE: runs experiments only; reports/figures are rebuilt on the laptop.
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N agentmem-grid
#$ -q gpu.q@gpu-1-1,gpu.q@gpu-1-4
#$ -l gpu=1
source /usr/local/gpuallocation.sh

PY=""
for CAND in /usr/bin/python38 /usr/bin/python3.8 /usr/local/bin/python3.8; do
  [ -x "$CAND" ] && "$CAND" -c "import lzma, ssl" 2>/dev/null && PY=$CAND && break
done
[ -z "$PY" ] && { echo "NO usable python3.8 on $(hostname)"; exit 1; }
echo "Using python: $PY"

export PATH="$HOME/.local/bin:$PATH"
CUPTI=$(ls -d /usr/local/cuda-12*/extras/CUPTI/lib64 2>/dev/null | tail -1)
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$CUPTI:$LD_LIBRARY_PATH"

$PY -m pip install --user --quiet pysqlite3-binary "urllib3<2"
mkdir -p ~/pyshim/pyairports
printf 'from pysqlite3.dbapi2 import *\nfrom pysqlite3 import dbapi2\n' > ~/pyshim/sqlite3.py
printf 'AIRPORT_LIST = []\n' > ~/pyshim/pyairports/airports.py
touch ~/pyshim/pyairports/__init__.py
export PYTHONPATH="$HOME/pyshim:$PYTHONPATH"
$PY -c "import sqlite3; print('sqlite3 shim OK', sqlite3.sqlite_version)" || exit 1

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=$((10000 + RANDOM % 20000))
export HPC_BASE_URL="http://localhost:$PORT/v1"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Port: $PORT"

$PY -c "import torch; print('torch OK', torch.__version__, 'cuda:', torch.cuda.is_available())" || exit 1

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --dtype float16 \
  --max-model-len 6144 --gpu-memory-utilization 0.92 --max-num-seqs 1 --enforce-eager \
  > vllm_server_$JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED — see vllm_server_$JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM is up."

# Full grid: every task x seed x all conditions (report step runs at home)
for TASK in custom_ds longmemeval_s; do
  for SEED in 1 2 3; do
    $PY scripts/run_experiment.py --task $TASK --all-conditions --seed $SEED
  done
done

kill $VLLM_PID
echo "GRID DONE — copy results/runs/*.jsonl + results/manifest.json home."
