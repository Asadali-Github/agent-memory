#!/usr/bin/env python
"""Pilot: small all-conditions grid on the custom task (and LongMemEval if the
data file is present), then print the headline summary table.

Mock mode by default (configs/base.yaml provider: mock) — validates the whole
pipeline on a CPU laptop with zero cost. Use it to fix budgets before real runs.

  python scripts/run_pilot.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentmem.logging.trajectory import load_records  # noqa: E402


def sh(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    sh("scripts/run_experiment.py", "--task", "custom_ds",
       "--all-conditions", "--seed", "1", "--limit", "3")
    if (ROOT / "data" / "longmemeval_s.json").exists():
        sh("scripts/run_experiment.py", "--task", "longmemeval_s",
           "--all-conditions", "--seed", "1", "--limit", "10")

    sys.path.insert(0, str(ROOT))
    from analysis.stats import failure_crosstab, records_to_df, summary_table

    for task in ["custom_ds", "longmemeval_s"]:
        recs = load_records(ROOT / "results" / "runs" / f"{task}.jsonl")
        if not recs:
            continue
        df = records_to_df(recs)
        print(f"\n=== {task}: headline summary ===")
        print(summary_table(df).to_string())
        print(f"\n=== {task}: failure tags (row-normalised) ===")
        print(failure_crosstab(df).to_string())


if __name__ == "__main__":
    main()
