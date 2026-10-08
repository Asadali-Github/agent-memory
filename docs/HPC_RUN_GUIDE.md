# Running the real experiment on university HPC

The pipeline talks to any OpenAI-compatible endpoint, so the HPC integration
is: start a vLLM server on a GPU node, point `.env` at it, flip one config
line. Nothing in the experiment code changes.

## 1. One-time setup (login node)

```bash
module avail python cuda        # find available modules (names vary by cluster)
module load python/3.11 cuda    # adjust to what `module avail` shows
python -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
pip install vllm
# Pull the model once (needs a HuggingFace token for Llama):
huggingface-cli login
```

Ask IT/supervisor for: the SLURM partition name for GPUs, max walltime,
and whether compute nodes have internet (if not, pre-download the model on
the login node with `HF_HOME` pointing at shared storage).

## 2. Serve the model (SLURM job)

`serve_llm.sbatch` (adjust partition/GPU/module names):

```bash
#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=gpu            # <- your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
module load python/3.11 cuda
source ~/venvs/vllm/bin/activate
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype auto
```

`sbatch serve_llm.sbatch`, then note the node it landed on (`squeue -u $USER`).

## 3. Point the pipeline at the server

If running the experiment on the SAME cluster (recommended — separate job or
interactive session):

```bash
# .env
HPC_BASE_URL=http://<node-name>:8000/v1
HPC_API_KEY=
```

From your laptop instead, tunnel first:
`ssh -L 8000:<node-name>:8000 <user>@<hpc-login-host>` then use
`HPC_BASE_URL=http://localhost:8000/v1`.

```yaml
# configs/base.yaml
provider: university_hpc
model: meta-llama/Llama-3.1-8B-Instruct   # must match the served model EXACTLY
```

Smoke test: `python scripts/run_experiment.py --task custom_ds --condition b --seed 1 --limit 1`

## 4. Pilot, freeze, full grid

1. Pilot (~30 min): `python scripts/run_experiment.py --task custom_ds --all-conditions --seed 1 --limit 3`
   plus 10 LongMemEval items per condition. Inspect results/report.md.
2. FREEZE budgets + model in configs/base.yaml; commit ("pilot freeze").
3. Full run: `python scripts/run_full_grid.py` (resumable — safe to requeue
   after walltime kills; completed cells are skipped).
4. Bring results home: copy `results/runs/*.jsonl` + `results/manifest.json`
   back into the repo; rerun `make_report.py`, `make_figures.py`,
   `export_for_eda.py` locally.

## 5. Embedding retriever (optional upgrade for M1/M4)

On HPC you can afford real embeddings:
`pip install sentence-transformers`, then in configs/conditions/m1.yaml and
m4.yaml set `retriever: embedding`. Record the choice — it must be identical
across all retrieval arms and reported in Ch3.

## Fallback if HPC falls through

configs/base.yaml `provider: groq` + `GROQ_API_KEY` in .env. The kill-switch
(budgets.gbp_cap) hard-stops spending; prices in src/agentmem/utils/costs.py.
