#!/bin/bash
# One-time setup on CERES (University of Essex). Run from ~/agent-memory:
#   bash scripts/hpc/setup_env.sh
# NOTE: the CERES login node kills processes after ~5 minutes. The big
# installs/downloads below RESUME if killed — just re-run this script
# until it prints ALL DONE.
set -e
cd "$(dirname "$0")/../.."

export PATH="$HOME/.local/bin:$PATH"

echo "== [1/4] Python packages (pip resumes from cache if killed) =="
# Versions PINNED for CERES: Python 3.9 + CentOS 7 (glibc 2.17, gcc 4.8).
# Newer releases have no prebuilt wheels for this platform and fail to build.
pip3.9 install --user --upgrade pip
pip3.9 install --user numpy==2.0.2 pandas==2.2.3 scipy==1.13.1 \
  statsmodels==0.14.4 matplotlib==3.9.4 python-dotenv pyyaml requests \
  pytest jupyter nbformat
pip3.9 install --user "vllm==0.6.6.post1" "huggingface_hub[cli]"

echo "== [2/4] Point config at the HPC model =="
sed -i 's/^provider: mock/provider: university_hpc/' configs/base.yaml
sed -i 's|^model: ""[^#]*|model: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"   |' configs/base.yaml
grep -E "^(provider|model):" configs/base.yaml

echo "== [3/4] Pre-download the model (~5.7GB, resumes if killed) =="
python3.9 -c "from huggingface_hub import snapshot_download; snapshot_download('hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4')"

echo "== [4/4] Sanity checks =="
python3.9 -m pytest tests/ -q || echo "WARNING: some tests failed — check before real runs"
ls data/longmemeval_s.json && echo "data OK"

echo "ALL DONE — next: qsub scripts/hpc/run_pilot_gpu.sh"
