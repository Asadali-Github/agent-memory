#!/usr/bin/env python
"""ONE COMMAND for the post-review Mode B re-runs (R1 + R2).

    python scripts/run_remediation.py --dry-run     # show the plan, touch nothing
    python scripts/run_remediation.py               # do it

WHAT IT DOES, AND WHY

R1 — re-run M2 on Mode B, all seeds.
    Two patches changed M2's behaviour, so every existing M2 Mode B record is
    void and must be regenerated:
      P1  summary.py now restores chronological prompt order (it previously
          returned turns newest-first and never re-sorted, so on histories
          below the fold threshold M2 was the baseline with reversed turns).
      P2  the fold threshold moved 1500 -> 800, so the summariser actually
          fires. At 1500 it fired on 0/258 items; at 800 it fires on 258/258.
    Because TrajectoryWriter resumes by (condition, item_id, seed, mode), the
    old M2 rows are dropped first — otherwise the re-run is a silent no-op.

R2 — run the b_reversed ordering-only control, all seeds.
    Identical turn selection to b, reverse-chronological prompt layout. Isolates
    the ordering effect that was previously confounded with M2, and pre-empts
    "how do you know M2's deficit isn't ordering?" in the viva.

NOT DONE HERE
    M3's Mode B re-run (R3) is optional — P3 changed only what is logged, not
    what the model does, so M3's success rates are unaffected. Add --with-m3 if
    you want true total_prompt_tokens for M3 rather than reconstructing cost
    analytically. Mode A (R4) is a separate, much longer job: finish M1 seeds 1
    and 3 before adding M2-M4, and keep the design balanced.

Every step is resumable and safe to interrupt. Drops are backed up to
results/runs/*.jsonl.<timestamp>.bak before anything is rewritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

RUNS = ROOT / "results" / "runs" / "custom_ds.jsonl"


def sh(args: list[str], dry: bool) -> None:
    print("+", "python", " ".join(args))
    if not dry:
        subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running or dropping anything")
    ap.add_argument("--with-m3", action="store_true",
                    help="also re-run M3 so total_prompt_tokens is measured, not "
                         "reconstructed (R3 — optional, see module docstring)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override seeds (default: seeds from configs/base.yaml)")
    ap.add_argument("--skip-report", action="store_true",
                    help="don't rebuild report/figures at the end")
    args = ap.parse_args()
    dry = args.dry_run

    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else config.get("seeds", [1])
    provider = config.get("provider", "?")

    rerun = ["m2"] + (["m3"] if args.with_m3 else [])

    print(f"\nprovider={provider}  seeds={seeds}  target={RUNS.relative_to(ROOT)}")
    if provider == "mock":
        print("\n!! provider is 'mock' — these runs would NOT be citable.\n"
              "   Set the real provider in configs/base.yaml before running.")
        if not dry:
            raise SystemExit(1)
    print(f"\nplan:\n  R1  drop + re-run {', '.join(c.upper() for c in rerun)} "
          f"on custom_ds, seeds {seeds}\n"
          f"  R2  run b_reversed (ordering control) on custom_ds, seeds {seeds}\n")

    # --- R1: clear the void rows, then regenerate them -----------------------
    for cond in rerun:
        sh(["scripts/drop_condition.py", "--file", str(RUNS), "--condition", cond], dry)
    for cond in rerun:
        for seed in seeds:
            sh(["scripts/run_experiment.py", "--task", "custom_ds",
                "--condition", cond, "--seed", str(seed)], dry)

    # --- R2: the ordering-only control arm -----------------------------------
    for seed in seeds:
        sh(["scripts/run_experiment.py", "--task", "custom_ds",
            "--condition", "b_reversed", "--seed", str(seed)], dry)

    if not args.skip_report:
        sh(["scripts/make_report.py"], dry)
        sh(["scripts/make_figures.py"], dry)

    print("\ndone." if not dry else "\ndry run — nothing was run or dropped.")
    print("Next: rewrite Ch4 §4.4 (real M2), §4.6 (ordering control), §4.3 "
          "(corrected token cost) and the abstract around the new numbers.")


if __name__ == "__main__":
    main()
