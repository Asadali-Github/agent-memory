# Runbook — filling the one missing Mode A record

**Gap:** `condition=b`, `seed=1`, `item_id=gpt4_78cf46a3` (Mode A).
Merged dataset holds 1,799 / 1,800 records. Baseline cell b/s1 has 119 of 120.

**Prepared:** 2026-08-03. Upload bundle: `hpc_upload/patch_b_s1_2026-08-03.zip`

---

## 0. What changed since the last plan (read this first)

The earlier advice said to re-run the whole 120-item `b × seed 1` cell. **That is not
necessary.** `TrajectoryWriter` (`src/agentmem/logging/trajectory.py`) is resumable: it
loads every `(condition, item_id, seed, mode)` key already present in the `--out` file,
and `run_mode_a()` skips those items. Pre-seed `--out` with the 119 existing records and
exactly one item runs.

This has been verified locally against the mock provider — the runner reported
`1 new records` and the file went 119 → 120 lines. (That mock output was written to a
temp path and deleted; nothing in `results/` was touched by the test.)

**No code changes are required.** The `--only-item` flag discussed earlier is unnecessary.

### Why the record is missing

`gpt4_78cf46a3` is **index 119 of 119** in the seed-1 item stream — the last item of the
cell. It is not an outlier: 505 turns, 501k characters, rank 98/120 by context size,
`question_type = temporal-reasoning`. The original job was cut off at the finish line
(wall clock or vLLM exit), not defeated by the item. A re-run should succeed.

---

## 1. Prerequisites on CERES

CERES uses **Grid Engine (`qsub`)**, not SLURM. Ignore `scripts/hpc/serve_llm.sbatch` —
it is marked OBSOLETE in its own header.

Confirm on the login node, in `~/agent-memory`:

```bash
grep -E "^(provider|model):" configs/base.yaml
# must print:
#   provider: university_hpc
#   model: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
```

If it still says `provider: mock`, apply step [2/4] of `scripts/hpc/setup_env.sh`:

```bash
sed -i 's/^provider: mock/provider: university_hpc/' configs/base.yaml
sed -i 's|^model: ""[^#]*|model: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"   |' configs/base.yaml
```

Also confirm `data/longmemeval_s.json` and the cached model are still present
(`HF_HUB_OFFLINE=1` is set by the job, so the model must already be in the HF cache).

Do **not** re-run the full `setup_env.sh` — its pip stage would race the existing
`~/.local` install. Only the sed step is needed.

---

## 2. Upload

From your laptop, in **PowerShell**. GlobalProtect VPN must show Connected first.

```powershell
scp "<path-to>\hpc_upload\patch_b_s1_2026-08-03.zip" <username>@ceres.essex.ac.uk:~/agent-memory/
```

Then log in:

```powershell
ssh -o ServerAliveInterval=60 <username>@ceres.essex.ac.uk
```

On CERES:

```bash
cd ~/agent-memory
unzip -o patch_b_s1_2026-08-03.zip
wc -l results/patch_b_s1/longmemeval_s_b_s1_resume.jsonl   # must be 119
chmod +x scripts/hpc/run_patch_b_s1.sh
```

The bundle contains only two files and overwrites nothing else:

| file | purpose |
|---|---|
| `scripts/hpc/run_patch_b_s1.sh` | the Grid Engine job |
| `results/patch_b_s1/longmemeval_s_b_s1_resume.jsonl` | 119 pre-seeded records (`--out` target) |

---

## 3. Run

```bash
cd ~/agent-memory
qsub scripts/hpc/run_patch_b_s1.sh
qstat -u <username>                    # r = running, qw = queued, empty = done
tail -n 40 amem-patch.o<JOB_ID>     # job log
```

Runtime is dominated by vLLM start-up: expect **5–15 minutes**, of which the actual
inference is one ingest pass plus a single generation.

The job self-checks and aborts rather than doing anything silently wrong. It refuses to
start if the resume file is missing, is not exactly 119 lines, already contains the item,
or if `base.yaml` still says `provider: mock`. It refuses to finish if the runner exits
non-zero or the file is not exactly 120 lines afterwards.

On success the log ends with:

```
records: 119 -> 120  (runner exit 0)
extracted 1 new record(s) -> results/patch_b_s1/patch_only.jsonl
PATCH DONE: b seed 1 item gpt4_78cf46a3
```

### If vLLM dies on start-up

Check `vllm_patch_b_s1_<JOB_ID>.log`. The usual cause is landing on a GTX1080Ti node —
the job already restricts to `gpu.q@gpu-1-1,gpu.q@gpu-1-4` (RTX2080) for exactly this
reason. Just resubmit. The job is idempotent: if a partial attempt somehow wrote the
record, the next submission's preflight will stop with "already in resume file".

---

## 4. Bring the record home

Back in **PowerShell** on your laptop (`exit` the ssh session first):

```powershell
scp <username>@ceres.essex.ac.uk:~/agent-memory/results/patch_b_s1/patch_only.jsonl "<path-to>\agent-memory\results\patch_b_s1\patch_only.jsonl"
```

Sanity-check it before it goes anywhere near the dataset:

```powershell
cd "<path-to>\agent-memory"
python scripts\verify_patch.py before
```

It checks the record's identity, that `prompt_version` is still `v1.0`, that the cost
fields are populated, and that `latency_ms` is plausible for vLLM (a mock-provider record
would come back sub-millisecond). Anything wrong and it exits non-zero with the reason —
do not merge until it prints `Safe to merge.`

---

## 5. Merge

Back up first, then merge with the patch **last** so it wins any key collision:

```powershell
Copy-Item results\rerun_2026_07_29\longmemeval_s_merged.jsonl `
          results\rerun_2026_07_29\longmemeval_s_merged.pre_patch.jsonl

python scripts\merge_runs.py `
  --input results\rerun_2026_07_29\longmemeval_s_merged.jsonl `
  --input results\patch_b_s1\patch_only.jsonl `
  --out   results\rerun_2026_07_29\longmemeval_s_merged.jsonl --dry-run
```

(The backtick `` ` `` is PowerShell's line continuation. Put nothing after it — not
even a space — or the line break breaks.)

The dry-run coverage table must show `b  120  120  120  360  COMPLETE` and
`1800 unique`. Only then re-run without `--dry-run`.

---

## 6. Resync everything downstream

Filling the cell moves b's Mode A n from 359 → 360 and its success rate off
`14/359 = 0.0390`. That propagates into every baseline contrast.

```powershell
python scripts\ch4_stats.py
python scripts\make_ch4_figures.py --outdir ..\writing\latex\figures
python scripts\make_report.py          # regenerates results\rerun_2026_07_29\report.md
```

Then edit by hand:

**`writing/latex/ch4_results.tex`**

| line | what to change |
|---|---|
| 6 | header comment `1799 records` → `1800` |
| 60 | **delete the disclosure sentence** in the Table 4.2 caption ("The baseline totals $359$ records rather than $360$ because …") — it becomes moot |
| 66 | B row: `$359$` → `$360$`, plus the refreshed rate and CI |
| 211 | Table caption: "all $359$ matched pairs" → `$360$` |
| 318 | "across all $359$ matched pairs" → `$360$` (and re-check $n_{01}=0$ still holds) |

Every other `359` in ch4 is the M3 inversion rate $\iota=0.359$ or the token
figure $+4{,}359$ — **do not touch those.**

**`results/audit_2026_08_02.md`** — lines 10, 75, 80, 82, 94, 122, 192, 203:
coverage 1,799 → 1,800 (100%), b row 119→120 / 359→360, drop the "1 item short"
status and the §6 missing-cell note, refresh the b success rate.

Add a short note recording that the gap was closed on this date and by which job ID,
rather than deleting the history outright.

**Compile:**

```powershell
cd "%USERPROFILE%\Desktop\Dissertation work\04_manuscript\latex"
pdflatex main.tex; pdflatex main.tex
```

---

## 7. Final verification

```powershell
cd "<path-to>\agent-memory"
python scripts\verify_patch.py after

Select-String -Path ..\writing\latex\ch4_results.tex -Pattern "359"
# every remaining hit should be M3's iota (0.359) or the token figure +4,359
```

`verify_patch.py after` confirms 1,800 records, all fifteen cells at 120, no duplicate
keys, and the item present — then prints the **new baseline success rate** so you know
exactly what to write into Table 4.2 and the audit report.

---

## If the HPC is unavailable

Leave the gap disclosed as it currently is. 1,799/1,800 with a named missing record and a
stated reason is a perfectly defensible position for a viva — the disclosure in the
Table 4.2 caption and audit report already does this correctly. **Do not** generate the
record with the mock provider to close the number; a mock record in a vLLM dataset is
fabrication, and it would be detectable from the token and latency columns.
