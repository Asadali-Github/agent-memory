#!/bin/bash
# FILE: run_patch_b_s1.sh — fill the ONE missing Mode A record:
#       condition=b, seed=1, item_id=gpt4_78cf46a3
#
# Why this is a 1-item job and not a 120-item job:
#   TrajectoryWriter (src/agentmem/logging/trajectory.py) is resumable — it
#   loads every (condition, item_id, seed, mode) key already present in the
#   --out file and run_mode_a() skips those items. We pre-seed --out with the
#   119 records that already exist, so exactly one item actually runs.
#   gpt4_78cf46a3 is index 119 of 119 in the seed-1 stream, i.e. the LAST item
#   of the cell — the original job was cut off at the finish line, not by
#   anything pathological about the item (it is rank 98/120 by context size).
#
# Environment is kept byte-identical to the original parallel Mode A jobs
# (scripts/hpc/run_one_gpu.sh): same python3.8, same AWQ-INT4 Llama-3.1-8B,
# same vLLM flags, temperature 0.0, seeded per-item via derive_seed().
#
# Submit with:  qsub scripts/hpc/run_patch_b_s1.sh
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N amem-patch
#$ -q gpu.q@gpu-1-1,gpu.q@gpu-1-4
#$ -l gpu=1
# ^ RTX2080 nodes ONLY — vLLM will not start on the GTX1080Ti nodes.
source /usr/local/gpuallocation.sh

# NOTE: deliberately NOT `set -u`. On CERES compute nodes PYTHONPATH and
# CUDA_VISIBLE_DEVICES may be unset, and the appends below would abort the job
# (job 2849509 died at line 68 this way). run_one_gpu.sh has no `set -u` either.
set +u

COND="b"
SEED="1"
ITEM="gpt4_78cf46a3"
RESUME="results/patch_b_s1/longmemeval_s_b_s1_resume.jsonl"

# ---- preflight -------------------------------------------------------------
if [ ! -f "$RESUME" ]; then
  echo "FATAL: $RESUME not found. Upload it before submitting this job."; exit 2
fi
N_BEFORE=$(grep -c . "$RESUME")
if [ "$N_BEFORE" -ne 119 ]; then
  echo "FATAL: expected 119 pre-seeded records, found $N_BEFORE. Aborting so we"
  echo "       do not silently re-run the whole cell."; exit 2
fi
if grep -q "$ITEM" "$RESUME"; then
  echo "FATAL: $ITEM is already in $RESUME — nothing to patch."; exit 2
fi
echo "Preflight OK: $N_BEFORE records pre-seeded, $ITEM absent."

grep -E "^(provider|model):" configs/base.yaml
if grep -q "^provider: mock" configs/base.yaml; then
  echo "FATAL: configs/base.yaml still says provider: mock. Run"
  echo "       bash scripts/hpc/setup_env.sh  (or apply its step [2/4] sed) first."
  exit 2
fi

# ---- interpreter (same discovery order as run_one_gpu.sh) ------------------
PY=""
for CAND in /usr/bin/python38 /usr/bin/python3.8 /usr/local/bin/python3.8; do
  [ -x "$CAND" ] && "$CAND" -c "import lzma, ssl" 2>/dev/null && PY=$CAND && break
done
[ -z "$PY" ] && { echo "NO usable python3.8 on $(hostname)"; exit 1; }
echo "Patch cell: $COND seed $SEED   node: $(hostname)   python: $PY"

export PATH="$HOME/.local/bin:$PATH"
CUPTI=$(ls -d /usr/local/cuda-12*/extras/CUPTI/lib64 2>/dev/null | tail -1)
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$CUPTI:$LD_LIBRARY_PATH"
export PYTHONPATH="$HOME/pyshim:$PYTHONPATH"
export HF_HUB_OFFLINE=1

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=$((10000 + RANDOM % 40000))
export HPC_BASE_URL="http://localhost:$PORT/v1"
echo "GPU: $CUDA_VISIBLE_DEVICES  Port: $PORT"

$PY -c "import torch; print('torch OK', 'cuda:', torch.cuda.is_available())" || exit 1

# ---- serve + run -----------------------------------------------------------
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  > vllm_patch_${COND}_s${SEED}_$JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED — see vllm_patch_${COND}_s${SEED}_$JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM is up."

$PY scripts/run_experiment.py \
  --task longmemeval_s --condition "$COND" --seed "$SEED" --out "$RESUME"
RC=$?

kill $VLLM_PID 2>/dev/null

# ---- postflight ------------------------------------------------------------
N_AFTER=$(grep -c . "$RESUME")
echo "records: $N_BEFORE -> $N_AFTER  (runner exit $RC)"
if [ "$RC" -ne 0 ] || [ "$N_AFTER" -ne 120 ]; then
  echo "PATCH FAILED — expected 120 records and exit 0."; exit 1
fi

# Isolate just the new record so the merge back home touches nothing else.
mkdir -p results/patch_b_s1
grep "\"item_id\": \"$ITEM\"" "$RESUME" > results/patch_b_s1/patch_only.jsonl
echo "extracted $(grep -c . results/patch_b_s1/patch_only.jsonl) new record(s) -> results/patch_b_s1/patch_only.jsonl"
echo "PATCH DONE: $COND seed $SEED item $ITEM"
