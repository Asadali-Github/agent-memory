#!/usr/bin/env python
"""ONE COMMAND reruns the entire experiment (FASER reproducibility requirement):

  python scripts/run_full_grid.py

For every seed in configs/base.yaml x every task with data x every condition:
run (resumable — completed cells are skipped), then rebuild the stats report,
the figures, and the EDA export. Safe to interrupt and rerun.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402


def sh(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    seeds = config.get("seeds", [1])
    tasks = ["custom_ds"]
    if (ROOT / "data" / "longmemeval_s.json").exists():
        tasks.append("longmemeval_s")
    else:
        print("NOTE: data/longmemeval_s.json missing -> skipping Mode A "
              "(see data/README.md for the download command)")

    for task in tasks:
        for seed in seeds:
            sh("scripts/run_experiment.py", "--task", task,
               "--all-conditions", "--seed", str(seed))

    sh("scripts/make_report.py")
    sh("scripts/make_figures.py")
    sh("scripts/export_for_eda.py")
    print("\nFull grid complete. See results/report.md, results/figures/, "
          "results/traces.jsonl")


if __name__ == "__main__":
    main()
