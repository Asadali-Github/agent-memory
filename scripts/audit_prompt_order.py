#!/usr/bin/env python
"""Audit the PROMPT ORDER actually used by each condition in a runs file.

    python scripts/audit_prompt_order.py --file results/rerun_2026_07_29/custom_ds_merged.jsonl

WHY THIS EXISTS
The expert review (B1) found that M2 presented its turns reverse-
chronologically while the baseline presented them chronologically, so the
measured "hierarchical summarisation harms performance" was really a measure of
prompt ordering. That patch is in. But the same question applies to every other
condition, and it had not been asked: if B is chronological and M1/M3/M4 are
score-ordered, then EVERY B-vs-Mx contrast mixes architecture with presentation,
not just the M2 one.

This script answers it from the logged provenance rather than from the code, so
it reports what the reported numbers were actually produced under.

Interpretation:
  chronological          same layout discipline as the baseline: clean contrast
  REVERSE-chronological  the M2 bug pattern: worth -11.9pp on Mode B
  score-order (mixed)    neither; ordering is a free variable in the contrast

A defensible write-up needs ordering declared as a design parameter in Ch3, the
b_reversed control arm quantifying what ordering alone is worth, and any arm
whose deficit is smaller than the ordering effect flagged as unresolved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def classify(steps: list[int]) -> str:
    if len(steps) < 2:
        return "<2 turns"
    if steps == sorted(steps):
        return "chronological"
    if steps == sorted(steps, reverse=True):
        return "REVERSE-chronological"
    return "score-order (mixed)"


def inversion_rate(steps: list[int]) -> float:
    """Fraction of turn pairs presented out of chronological order (0 = perfect)."""
    n = len(steps)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if steps[i] > steps[j])
    return inv / (n * (n - 1) / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/rerun_2026_07_29/custom_ds_merged.jsonl")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    kinds: dict[str, Counter] = defaultdict(Counter)
    inversions: dict[str, list[float]] = defaultdict(list)

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prov = rec.get("provenance") or []
        steps = [p["step"] for p in prov
                 if isinstance(p, dict) and p.get("step") is not None and p["step"] != -1]
        cond = rec.get("condition", "?")
        kinds[cond][classify(steps)] += 1
        if len(steps) >= 2:
            inversions[cond].append(inversion_rate(steps))

    cols = ["chronological", "REVERSE-chronological", "score-order (mixed)", "<2 turns"]
    print(f"\nprompt-order audit — {path.name}\n")
    print(f"{'cond':<8}" + "".join(f"{c:>24}" for c in cols) + f"{'inversion':>12}{'n':>7}")
    for cond in sorted(kinds):
        c = kinds[cond]
        inv = inversions[cond]
        mean_inv = sum(inv) / len(inv) if inv else 0.0
        print(f"{cond:<8}" + "".join(f"{c[k]:>24}" for k in cols)
              + f"{mean_inv:>12.3f}{sum(c.values()):>7}")

    print("\ninversion = mean fraction of turn pairs shown out of chronological order")
    print("  0.00 = identical layout discipline to the baseline")
    print("  1.00 = fully reversed (the pre-patch M2 pattern, worth -11.9pp)")
    print("\nAny condition with a non-zero inversion rate is contrasted against a")
    print("chronological baseline, so its effect is architecture + ordering.")


if __name__ == "__main__":
    main()
