#!/usr/bin/env python
"""Judge validation protocol (reported in Ch3): is the automated grader
trustworthy enough to carry the headline metric?

Two-phase workflow:
  1. SAMPLE: draw a stratified sample of graded records and write
     results/judge_validation_sample.csv with an empty `human_label` column.
       python scripts/validate_judge.py sample --n 50
     -> fill human_label with 1 (correct) / 0 (incorrect), blind to the
        judge's verdict (the judge column is withheld in a separate key file).
  2. SCORE: compute agreement + Cohen's kappa between human and judge.
       python scripts/validate_judge.py score
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from agentmem.logging.trajectory import load_records  # noqa: E402

SAMPLE_CSV = ROOT / "results" / "judge_validation_sample.csv"
KEY_CSV = ROOT / "results" / "judge_validation_key.csv"  # judge verdicts, kept apart


def sample(n: int, seed: int) -> None:
    records = []
    for f in sorted((ROOT / "results" / "runs").glob("*.jsonl")):
        records.extend(load_records(f))
    if not records:
        sys.exit("No run records to sample from.")
    rng = random.Random(seed)
    succ = [r for r in records if r["success"]]
    fail = [r for r in records if not r["success"]]
    rng.shuffle(succ), rng.shuffle(fail)
    chosen = succ[: n // 2] + fail[: n - n // 2]
    rng.shuffle(chosen)

    blind = pd.DataFrame([{
        "run_id": r["run_id"], "question": r["question"],
        "gold": r["gold"], "prediction": r["prediction"], "human_label": "",
    } for r in chosen])
    key = pd.DataFrame([{"run_id": r["run_id"], "judge_label": int(r["success"])}
                        for r in chosen])
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(SAMPLE_CSV, index=False)
    key.to_csv(KEY_CSV, index=False)
    print(f"Wrote {len(blind)} rows -> {SAMPLE_CSV}")
    print(f"Fill `human_label` (1/0) WITHOUT looking at {KEY_CSV.name}, then run:")
    print("  python scripts/validate_judge.py score")


def score() -> None:
    blind = pd.read_csv(SAMPLE_CSV)
    key = pd.read_csv(KEY_CSV)
    df = blind.merge(key, on="run_id")
    df = df.dropna(subset=["human_label"])
    if df.empty:
        sys.exit("human_label column is still empty.")
    h = df["human_label"].astype(int)
    j = df["judge_label"].astype(int)
    agree = float((h == j).mean())
    po = agree
    pe = float((h.mean() * j.mean()) + ((1 - h.mean()) * (1 - j.mean())))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"n={len(df)}  raw agreement={agree:.3f}  Cohen's kappa={kappa:.3f}")
    print("Disagreements (inspect + discuss in Ch3):")
    for _, row in df[h != j].iterrows():
        print(f"  {row.run_id}: judge={row.judge_label} human={int(row.human_label)} "
              f"| gold={row.gold!r} pred={str(row.prediction)[:60]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=1)
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.n, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
