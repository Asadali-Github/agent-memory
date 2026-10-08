#!/bin/bash
# FILE: run_remaining_jobs.sh — everything still outstanding after the
# 28 July corrections, in priority order, on ONE GPU allocation.
#
#   qsub scripts/hpc/run_remaining_jobs.sh          # all tiers
#   qsub scripts/hpc/run_remaining_jobs.sh 1        # tier 1 only
#
# MEASURED COST (from logged latency in the existing runs, not guessed):
#   Mode B ~3.2 min per condition per seed; Mode A ~4.4 min. Whole file < 2 h.
#
# BEFORE SUBMITTING — three things that will silently ruin the run:
#
#  1. SYNC THE FIXED CODE. harness/runner.py previously logged
#     ctx_info["provenance"][:20], truncating provenance AFTER sorting it.
#     Because B sorts ascending and the reverse-chronological arms sort
#     descending, that made the two look like they selected disjoint turns and
#     it corrupted every inversion-rate figure. If the HPC checkout still has
#     the old line, every ordering number this run produces is worthless.
#     Verify on the cluster:  grep -n 'provenance"\]\[:20\]' src/agentmem/harness/runner.py
#     That must return NOTHING.
#
#  2. RESUME IS BY (condition, item_id, seed, mode). Re-running a condition
#     that already has rows is a SILENT NO-OP. Tiers 1 and 2 re-run b and m2,
#     whose old rows must be dropped first — this script writes to a fresh
#     directory instead, so the archived Jul-20 data is never touched.
#
#  3. NOT MOCK. run_experiment refuses nothing in mock mode, so confirm
#     configs/base.yaml has provider: university_hpc and the model set.
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N amem-remaining
#$ -q gpu.q@gpu-1-1,gpu.q@gpu-1-4
#$ -l gpu=1
source /usr/local/gpuallocation.sh

TIER="${1:-all}"

PY=""
for CAND in /usr/bin/python38 /usr/bin/python3.8 /usr/local/bin/python3.8; do
  [ -x "$CAND" ] && "$CAND" -c "import lzma, ssl" 2>/dev/null && PY=$CAND && break
done
[ -z "$PY" ] && { echo "NO usable python3.8 on $(hostname)"; exit 1; }

# --- guard: refuse to run against the truncating runner -----------------
if grep -q 'provenance"\]\[:20\]' src/agentmem/harness/runner.py; then
  echo "ABORT: runner.py still truncates provenance to 20 entries."
  echo "       Sync the fixed code before running — see header note 1."
  exit 1
fi
echo "provenance fix: PRESENT"

# --- guard: refuse to run in mock mode -------------------------------------
# The laptop dev default is `provider: mock`. If a code upload overwrites
# configs/base.yaml, the grid will run happily and produce meaningless
# results, because mock mode is a valid provider rather than an error.
PROVIDER=$(grep -E '^provider:' configs/base.yaml | awk '{print $2}')
MODELCFG=$(grep -E '^model:' configs/base.yaml | cut -d: -f2- | tr -d ' "')
if [ "$PROVIDER" = "mock" ] || [ -z "$MODELCFG" ]; then
  echo "ABORT: configs/base.yaml has provider='$PROVIDER' model='$MODELCFG'."
  echo "       Mock mode produces meaningless results. Restore the frozen config:"
  echo "         sed -i 's|^provider: mock|provider: university_hpc|' configs/base.yaml"
  echo "         sed -i 's|^model: .*|model: \"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4\"|' configs/base.yaml"
  exit 1
fi
echo "provider: $PROVIDER   model: $MODELCFG"

export PATH="$HOME/.local/bin:$PATH"
CUPTI=$(ls -d /usr/local/cuda-12*/extras/CUPTI/lib64 2>/dev/null | tail -1)
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$CUPTI:$LD_LIBRARY_PATH"
$PY -m pip install --user --quiet pysqlite3-binary "urllib3<2"
mkdir -p ~/pyshim/pyairports
printf 'from pysqlite3.dbapi2 import *\nfrom pysqlite3 import dbapi2\n' > ~/pyshim/sqlite3.py
printf 'AIRPORT_LIST = []\n' > ~/pyshim/pyairports/airports.py
export PYTHONPATH="$HOME/pyshim:$PWD/src:$PYTHONPATH"

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=$((8000 + RANDOM % 500))
export HPC_BASE_URL="http://localhost:$PORT/v1"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  > vllm_remaining_$JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED"; exit 1; }
  sleep 10
done
echo "vLLM is up."

# Fresh directory: the archived Jul-20 results are left untouched, so nothing
# already reported can be silently overwritten.
OUT=results/runs/rerun_2026_07
mkdir -p $OUT

run () {  # run <task> <condition> <seed>
  echo ">>> $1 / $2 / seed $3"
  $PY scripts/run_experiment.py --task "$1" --condition "$2" --seed "$3" \
      --out "$OUT/$1.jsonl"
}

# ---------------------------------------------------------------------------
# TIER 1 — REQUIRED. Section 4.6 currently has no data behind it.
# b_reversed has zero scored records; the ordering result is presently carried
# by the relabelled pre-patch M2 arm. b is re-run alongside it so the paired
# McNemar contrast comes from one code version.
# NOTE: b_reversed is deliberately excluded from --all-conditions (it is a
# control arm, not a registered architecture), so it needs --condition.
# Cost: 6 cells x ~3.2 min = ~20 min.
# ---------------------------------------------------------------------------
if [ "$TIER" = "all" ] || [ "$TIER" = "1" ]; then
  for SEED in 1 2 3; do
    run custom_ds b          $SEED
    run custom_ds b_reversed $SEED
  done
fi

# ---------------------------------------------------------------------------
# TIER 2 — REQUIRED for any claim about hierarchical summarisation.
# At the old 1500-token threshold the summariser fired on 0/258 items, so the
# archived M2 measures ordering, not summarisation. At 800 it fires on 258/258.
# Without this the dissertation has NO measurement of a MemGPT-style memory.
# Cost: 3 cells x ~3.2 min = ~10 min.
# ---------------------------------------------------------------------------
if [ "$TIER" = "all" ] || [ "$TIER" = "2" ]; then
  for SEED in 1 2 3; do run custom_ds m2 $SEED; done
fi

# ---------------------------------------------------------------------------
# TIER 3 — repairs the single largest claim in the dissertation.
# Mode A M1 (0.032 -> 0.433, a 13x lift) currently rests on ONE seed, and the
# baseline's seed 3 died at 102/120 items. Two seeds of M1 plus the baseline
# tail make it a three-seed result.
# Cost: ~15 min.
# ---------------------------------------------------------------------------
if [ "$TIER" = "all" ] || [ "$TIER" = "3" ]; then
  run longmemeval_s m1 1
  run longmemeval_s m1 3
  run longmemeval_s b  3
fi

# ---------------------------------------------------------------------------
# TIER 4 — OPTIONAL, in descending value.
#
# (a) Mitigation experiment. M1/M4 re-run with chronological_order enabled:
#     identical selection, identical token count, chronological layout. Turns
#     the "one-line fix removes the ordering tax" argument from a proposal into
#     a measurement. Requires m1_chrono.yaml / m4_chrono.yaml (clone m1/m4 and
#     add `chronological_order: true`).
# (b) M3 metered, so total_prompt_tokens is measured rather than reconstructed,
#     collapsing the +8.2%-+17.3% bracket to one number.
# (c) Mode A m2/m3/m4 — completes the matrix. Lowest priority: a balanced
#     two-condition Mode A is more defensible than five ragged ones.
# Cost: ~60 min for all three.
# ---------------------------------------------------------------------------
if [ "$TIER" = "all" ] || [ "$TIER" = "4" ]; then
  for SEED in 1 2 3; do
    run custom_ds m1_chrono $SEED
    run custom_ds m4_chrono $SEED
    run custom_ds m3        $SEED
  done
  for SEED in 1 2 3; do
    for C in m2 m3 m4; do run longmemeval_s $C $SEED; done
  done
fi

kill $VLLM_PID
echo
echo "DONE. Copy $OUT/*.jsonl home, then on the laptop run:"
echo "  python scripts/audit_prompt_order.py --file $OUT/custom_ds.jsonl"
echo "  python scripts/ordering_decomposition.py --file $OUT/custom_ds.jsonl"
echo "  python scripts/make_report.py && python scripts/make_figures.py"
