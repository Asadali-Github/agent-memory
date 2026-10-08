#!/bin/bash
# FILE: setup_gpu_job.sh — one-time GPU-side setup, run AS A JOB (GPU nodes
# have internet; login node kills long processes). Submit from ~/agent-memory:
#   qsub scripts/hpc/setup_gpu_job.sh
# Takes 10–25 min. Check output: cat setup-gpu.o*   (ends with SETUP_DONE)
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N setup-gpu
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh

PY=/usr/local/bin/python3.8
[ -x "$PY" ] || PY=$(command -v python3.8 || command -v python38)
echo "Using python: $PY ($($PY --version 2>&1))"
export PIP_NO_CACHE_DIR=1

echo "== disk quota before =="; quota -s 2>/dev/null | head -5; df -h $HOME | tail -1

echo "== [1/4] pip =="
$PY -m pip install --user --upgrade pip

echo "== [2/4] harness deps (Python 3.8 compatible pins) =="
$PY -m pip install --user "numpy==1.24.4" pyyaml requests python-dotenv pytest

echo "== [3/4] vLLM 0.6.2 (prebuilt for this platform: CentOS7 + py3.8) =="
$PY -m pip install --user "vllm==0.6.2"

echo "== [4/4] model download (~5.7GB) + config =="
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download('hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4')
print("model downloaded OK")
PYEOF

sed -i 's/^provider: mock/provider: university_hpc/' configs/base.yaml
sed -i 's|^model: ""[^#]*|model: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"   |' configs/base.yaml
grep -E "^(provider|model):" configs/base.yaml

echo "== disk quota after =="; quota -s 2>/dev/null | head -5; df -h $HOME | tail -1
echo SETUP_DONE
