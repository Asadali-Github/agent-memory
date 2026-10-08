# CERES quick reference — running the dissertation experiments

Cluster: `ceres.essex.ac.uk` (University of Essex CERES, Grid Engine).
Login: Essex username (<username>) + password + Microsoft push MFA.
VPN: GlobalProtect must be Connected (it usually already is).

## ✅ STATUS 19 Jul, 20:23 — PILOT SUCCEEDED (job 2847608)
All 5 conditions x 3 items ran on gpu-1-1 (RTX2080) with vLLM + AWQ Llama.
Results: `~/agent-memory/results/runs/custom_ds.jsonl` (140 records).
Working scripts (both synced to this repo): scripts/hpc/run_pilot_gpu.sh,
scripts/hpc/run_grid_gpu.sh. The grid script on the CLUSTER must be updated
to this repo's version before the full run (copy the file contents in via
nano).

Next steps:
1. Copy pilot records home + eyeball them (few answers per condition).
2. Freeze configs/base.yaml (budgets+model), commit ("pilot freeze").
3. Update cluster's run_grid_gpu.sh to repo version → qsub it.
   Full grid = 2 tasks x 3 seeds x 5 conditions; hours-to-days on one GPU;
   resumable — re-qsub if killed.
4. Copy results/runs/*.jsonl + manifest home → make_report/figures/EDA.

## Old status (18 Jul) — superseded
Done: repo + data on cluster ✓, vLLM 0.6.2 + torch installed ✓, model
downloaded ✓, GPU verified (torch cuda: True) ✓. Remaining: submit the
FIXED pilot script (in `scripts/hpc/run_pilot_gpu.sh` in this folder — it
fixes a missing lzma module by choosing the right Python). To resume:
1. VPN connected? (GlobalProtect → Connected)
2. `ssh -o ServerAliveInterval=60 <username>@ceres.essex.ac.uk`
3. Copy the whole content of scripts/hpc/run_pilot_gpu.sh, then on ceres run:
   `nano ~/agent-memory/scripts/hpc/run_pilot_gpu.sh`, replace contents,
   save) — then: `cd ~/agent-memory && qsub scripts/hpc/run_pilot_gpu.sh`
4. Watch: `qstat -u <username>`, then `tail -n 30 ~/agent-memory/agentmem-pilot.o<jobid>`
5. Success = "vLLM is up." then "PILOT DONE".

## The workflow (original plan)

**1. Connect** (done — repeat whenever the connection drops):
```
ssh <username>@ceres.essex.ac.uk
```

**2. Files uploaded** (done): code.zip unpacked to `~/agent-memory`,
datasets moved into `~/agent-memory/data/`.

**3. One-time setup — NEXT STEP.** The login node has no pip and kills
long processes, so first hop onto a compute node:
```
qrsh
cd ~/agent-memory && bash scripts/hpc/setup_env.sh
```
Wait for `ALL DONE` (15–30 min: installs vLLM + downloads the 5.7GB model).
If it fails partway, re-run the `bash ...` line — it resumes.
Type `exit` to leave qrsh when finished.

**4. Pilot run** (from the login node, in `~/agent-memory`):
```
qstat -g c          # check queue load first (optional)
qsub scripts/hpc/run_pilot_gpu.sh
```
Monitor:
```
qstat -u <username>    # r = running, qw = waiting. Empty = finished
```
Output appears in `~/agent-memory/agentmem-pilot.o<jobid>`; read it with
`cat agentmem-pilot.o*`. Pilot = all 5 memory conditions × 3 items.

**5. Inspect pilot → freeze config → full grid:**
```
qsub scripts/hpc/run_grid_gpu.sh
```
Resumable: if it dies or hits a time limit, `qsub` it again — completed
cells are skipped.

**6. Bring results home** (from your laptop, PowerShell):
```
scp -r <username>@ceres.essex.ac.uk:~/agent-memory/results/runs "<path-to>\agent-memory\results\"
scp <username>@ceres.essex.ac.uk:~/agent-memory/results/manifest.json "<path-to>\agent-memory\results\"
```
Then locally run `make_report.py`, `make_figures.py`, `export_for_eda.py`.

## Key decisions made (record in Ch3 methodology)
- Model: **hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4** (4-bit AWQ
  quantization) served by vLLM — chosen because CERES GPUs are 8–11GB and
  the full fp16 8B model (~16GB) does not fit on one card.
- GPU: single **RTX2080** (jobs pinned to nodes gpu-1-1 / gpu-1-4; the
  GTX1080Ti nodes are compute capability 6.1, below vLLM's 7.0 minimum).
- Server and experiment run in the SAME batch job (CERES GPU queue is
  batch-only), talking over localhost:8000.
- `gpu=1` requested → shortest queue wait.

## Gotchas learned the hard way
- Commands with a server name go on ONE line; Enter only at the end.
- Never type your password into the prompt — wait for `password:`.
- `pip3.9`/`python3.9` exist only on compute nodes (use `qrsh`).
- Login-node processes are killed after ~5 min.
- Right-click pastes into the terminal.
