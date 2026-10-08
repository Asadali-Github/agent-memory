# agent-memory

Code, results and figures for my MSc Data Science dissertation at the University of Essex (MA981, 2026):

**What Should an Agent Remember? A Comparative Study of Memory Architectures for Long-Horizon LLM Agents**

## What this project tests

An LLM agent working on a long task can't keep everything in its prompt, so it needs some kind of memory. This project compares five memory designs on one fixed open-weight model, run on the University of Essex CERES HPC cluster.

Every condition gets the same token budget for reading its memory (the Equal-Budget Principle), so the designs differ only in what they choose to put in that space. All five sit behind one locked interface in `src/agentmem/memory/base.py`.

| Condition | Class            | Idea                                            |
|-----------|------------------|-------------------------------------------------|
| B         | `NoMemory`       | recency window (baseline)                       |
| M1        | `EpisodicMemory` | verbatim turns, relevance retrieval             |
| M2        | `SummaryMemory`  | hierarchical running summary                    |
| M3        | `SemanticMemory` | LLM-extracted fact store, update on conflict    |
| M4        | `DecayMemory`    | relevance retrieval with exponential recency decay |

Four layout control arms keep the same memory content but change its order: `b_reversed`, `m1_chrono`, `m3_chrono` and `m4_chrono`.

There are two evaluations:

- **Mode A, memory probing:** a stratified subset of LongMemEval, 360 items per condition.
- **Mode B, memory in use:** a custom multi-step project-tracker task, 219 items per condition over three seeds.

That gives 3,771 experiment records in total (1,800 in Mode A and 1,971 in Mode B).

## Main results

Success rate in Mode B, from `results/rerun_2026_07_29/ch4_stats_output_2026-08-04.txt`:

| Condition    | Success |
|--------------|---------|
| B (baseline) | 0.744   |
| B reversed   | 0.626   |
| M1           | 0.836   |
| M1 chrono    | 0.973   |
| M2           | 0.913   |
| M3           | 0.753   |
| M4           | 0.740   |
| M4 chrono    | 0.973   |

The result that surprised me was about order, not architecture. Putting the same retrieved memories in chronological order lifted M1 from 0.836 to 0.973 and M4 from 0.740 to 0.973. Reversing the baseline's order dropped it from 0.744 to 0.626. In this setup, how the memory was laid out in the prompt mattered about as much as which memory design produced it.

In Mode A, episodic memory (M1) did best at 0.408, against 0.039 for the baseline.

Contrasts use exact McNemar tests with Holm correction and bootstrap confidence intervals. The full statistics output is in `results/rerun_2026_07_29/`, produced by `scripts/ch4_stats.py`.

## Quickstart (CPU laptop, no cost)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # test suite, all invariants
python scripts/run_pilot.py         # small all-conditions grid, mock LLM
```

## The full run

1. Get the data: see `data/README.md` (LongMemEval download).
2. Point the code at a model: see `docs/HPC_RUN_GUIDE.md` (vLLM on HPC, with Groq as a fallback). Copy `.env.example` to `.env` and fill in your own keys.
3. Run `python scripts/run_full_grid.py`. It is resumable and writes the report, figures and traces into `results/`.

You can also run `Run_Experiments.ipynb` from top to bottom.

## Layout

```
src/agentmem/   memory/ (5 conditions + locked interface)  llm/  harness/  tasks/
                eval/ (recall, judge, failures)  logging/  utils/
configs/        base.yaml (single source of truth) + conditions/ + tasks/
analysis/       stats.py (McNemar + Holm, bootstrap CI, logit, crosstabs)
scripts/        experiment drivers, statistics, figures, HPC job scripts
tests/          pytest suite, including an official-schema LongMemEval fixture
results/        run records, statistics outputs and figures
docs/           HPC run guide and run notes
```

Reproducibility: fixed seeds, config-driven constants, resumable JSONL with a run id on every record, and a manifest saved with each run.

## Data

LongMemEval is not included here. Download it from the official release at https://github.com/xiaowu0162/LongMemEval and follow `data/README.md`.

## Notes

- This is the code as submitted for the dissertation in August 2026, made public after submission with my supervisor's agreement. The history starts fresh at this version.
- Before publishing I removed personal login details and local file paths from the run notes in `docs/` and two log files, and made `results/audit_2026_08_04/verify_claims.py` use a relative path. No experiment code or result changed.
- A few older files in `results/` are marked SUPERSEDED at the top. The numbers in the dissertation come from `results/rerun_2026_07_29/`.

## Author

Asad Ali, MSc Data Science, University of Essex
GitHub: [Asadali-Github](https://github.com/Asadali-Github) · LinkedIn: [asad-ali-2a3210177](https://www.linkedin.com/in/asad-ali-2a3210177)
