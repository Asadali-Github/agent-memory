#!/bin/bash
# FILE: run_one_gpu.sh — run ONE (condition, seed) cell of Mode A on its own
# GPU, writing to its OWN output file so parallel jobs never collide and the
# main grid job (2847759) is completely unaffected.
#
# Submit like:  qsub scripts/hpc/run_one_gpu.sh m3 1
#               qsub scripts/hpc/run_one_gpu.sh m3 2   ... etc.
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N amem-par
#$ -q gpu.q@gpu-1-1,gpu.q@gpu-1-4
#$ -l gpu=1
# ^ RTX2080 nodes ONLY. vLLM's engine will not start on the GTX1080Ti nodes
#   (compute capability too low) — that is what killed the first batch.
source /usr/local/gpuallocation.sh

COND="$1"          # b | m1 | m2 | m3 | m4
SEED="$2"          # 1 | 2 | 3
if [ -z "$COND" ] || [ -z "$SEED" ]; then
  echo "usage: qsub run_one_gpu.sh <condition> <seed>"; exit 2
fi

PY=""
for CAND in /usr/bin/python38 /usr/bin/python3.8 /usr/local/bin/python3.8; do
  [ -x "$CAND" ] && "$CAND" -c "import lzma, ssl" 2>/dev/null && PY=$CAND && break
done
[ -z "$PY" ] && { echo "NO usable python3.8 on $(hostname)"; exit 1; }
echo "Cell: $COND seed $SEED   node: $(hostname)   python: $PY"

export PATH="$HOME/.local/bin:$PATH"
CUPTI=$(ls -d /usr/local/cuda-12*/extras/CUPTI/lib64 2>/dev/null | tail -1)
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$CUPTI:$LD_LIBRARY_PATH"

# shims already installed + created by the main runs; DON'T pip-install here
# (six concurrent --user installs would race on the shared ~/.local). Just
# point at the existing pyshim and cached model.
export PYTHONPATH="$HOME/pyshim:$PYTHONPATH"
export HF_HUB_OFFLINE=1

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=$((10000 + RANDOM % 40000))
export HPC_BASE_URL="http://localhost:$PORT/v1"
echo "GPU: $CUDA_VISIBLE_DEVICES  Port: $PORT"

$PY -c "import torch; print('torch OK', 'cuda:', torch.cuda.is_available())" || exit 1

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  > vllm_par_${COND}_s${SEED}_$JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED — see vllm_par_${COND}_s${SEED}_$JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM is up."

# Own output file — no collision with the main job's longmemeval_s.jsonl
mkdir -p results/runs/parallel
OUT="results/runs/parallel/longmemeval_s_${COND}_s${SEED}.jsonl"
$PY scripts/run_experiment.py --task longmemeval_s --condition "$COND" --seed "$SEED" --out "$OUT"

kill $VLLM_PID
echo "CELL DONE: $COND seed $SEED -> $OUT"
