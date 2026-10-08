"""Merge parallel per-cell JSONL files into one runs file, deduping on record_key.

The parallel jobs (scripts/hpc/run_one_gpu.sh) write to their own files so they
never collide. Some cells also exist in the main grid file, so a naive `cat`
double-counts them into every rate you compute. This merges on the same key
TrajectoryWriter uses -- (condition, item_id, seed, mode) -- and reports what
came from where.

Precedence: later --input files win on a key collision. Put the file you trust
most LAST (normally the patched/most recent run).

Usage:
    python scripts/merge_runs.py \
        --input results/runs/longmemeval_s.jsonl \
        --input "results/runs/parallel/longmemeval_s_*.jsonl" \
        --out   results/runs/longmemeval_s_merged.jsonl

    # dry run first -- prints the coverage table, writes nothing
    python scripts/merge_runs.py --input ... --out ... --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_PER_CELL = 120  # Mode A items per (condition, seed)


def record_key(rec: dict) -> tuple:
    return (rec["condition"], rec["item_id"], rec["seed"], rec["mode"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="file or glob; repeatable; later inputs win collisions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", type=int, default=EXPECTED_PER_CELL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.input:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"WARNING: no files match {pattern!r}")
        paths += [Path(h) for h in hits]

    merged: dict[tuple, dict] = {}
    source: dict[tuple, str] = {}
    overwritten = Counter()
    per_file = Counter()

    for p in paths:
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                k = record_key(rec)
            except (json.JSONDecodeError, KeyError):
                continue
            if k in merged and source[k] != p.name:
                overwritten[f"{source[k]} -> {p.name}"] += 1
            merged[k] = rec
            source[k] = p.name
            n += 1
        per_file[p.name] = n
        print(f"  read {n:5d} records from {p.name}")

    print(f"\n{sum(per_file.values())} records read, {len(merged)} unique "
          f"({sum(per_file.values()) - len(merged)} duplicates collapsed)")
    if overwritten:
        print("\nCollisions resolved (later input won):")
        for pair, c in overwritten.most_common():
            print(f"  {c:5d}  {pair}")

    # Coverage table
    cells: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for rec in merged.values():
        cells[rec["condition"]][rec["seed"]] += 1

    print(f"\nCoverage (expected {args.expected} per cell):")
    seeds = sorted({s for c in cells.values() for s in c})
    print("  cond  " + "".join(f"{'s'+str(s):>8}" for s in seeds) + f"{'total':>9}  status")
    for cond in sorted(cells):
        row = cells[cond]
        total = sum(row.values())
        full = args.expected * len(seeds)
        status = "COMPLETE" if total >= full else f"{100*total/full:.0f}%"
        print("  " + f"{cond:<6}" + "".join(f"{row.get(s, 0):>8}" for s in seeds)
              + f"{total:>9}  {status}")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for k in sorted(merged, key=lambda k: (k[0], k[2], str(k[1]))):
            f.write(json.dumps(merged[k], ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {len(merged)} records -> {out}")


if __name__ == "__main__":
    main()
