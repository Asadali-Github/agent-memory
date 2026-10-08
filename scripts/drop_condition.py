"""Remove all records for one condition from a runs JSONL so it can be re-run.

TrajectoryWriter resumes by skipping (condition, item_id, seed, mode) keys that
already exist in the file. Re-running a condition without clearing its rows is a
silent no-op: the job starts, writes nothing, and exits successfully.

Usage:
    python scripts/drop_condition.py --file results/runs/custom_ds.jsonl --condition m2
    python scripts/drop_condition.py --file results/runs/custom_ds.jsonl --condition m2 --seed 1

Writes a timestamped .bak alongside the original before touching anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="results/runs/*.jsonl")
    ap.add_argument("--condition", required=True, help="e.g. m2")
    ap.add_argument("--seed", type=int, default=None,
                    help="drop only this seed (default: all seeds)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        match = rec.get("condition") == args.condition and (
            args.seed is None or rec.get("seed") == args.seed
        )
        if match:
            dropped += 1
        else:
            kept.append(line)

    scope = f"{args.condition}" + (f" seed {args.seed}" if args.seed is not None else " (all seeds)")
    print(f"{path.name}: {dropped} records match {scope}, {len(kept)} kept")

    if args.dry_run:
        print("dry run — nothing written")
        return
    if dropped == 0:
        print("nothing to do")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".jsonl.{stamp}.bak")
    shutil.copy2(path, backup)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"backup: {backup.name}\nrewrote: {path.name}")


if __name__ == "__main__":
    main()
