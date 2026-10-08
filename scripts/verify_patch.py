#!/usr/bin/env python
"""Checks for the b/seed-1 missing-record patch. Avoids fighting PowerShell quoting.

    python scripts/verify_patch.py before    # inspect the downloaded patch record
    python scripts/verify_patch.py after     # verify the merged dataset is 1800/1800
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "rerun_2026_07_29" / "longmemeval_s_merged.jsonl"
PATCH = ROOT / "results" / "patch_b_s1" / "patch_only.jsonl"
ITEM = "gpt4_78cf46a3"


def load(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"FAIL: {p} not found")
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def before() -> None:
    recs = load(PATCH)
    if len(recs) != 1:
        sys.exit(f"FAIL: expected exactly 1 record in {PATCH.name}, got {len(recs)}")
    r = recs[0]
    got = (r["condition"], r["seed"], r["mode"], r["item_id"])
    want = ("b", 1, "A", ITEM)
    if got != want:
        sys.exit(f"FAIL: record identity {got} != {want}")
    if r.get("prompt_version") != "v1.0":
        sys.exit(f"FAIL: prompt_version is {r.get('prompt_version')!r}, expected 'v1.0' — "
                 "the prompt changed, this record is not comparable to the other 1,799")
    if r.get("gbp_spent") is None or r.get("prompt_tokens", 0) <= 0:
        sys.exit("FAIL: cost fields look unpopulated")
    # A mock-provider record is detectable: mock latency is sub-millisecond.
    if float(r.get("latency_ms", 0)) < 50:
        sys.exit(f"FAIL: latency_ms={r['latency_ms']} is implausibly low for vLLM — "
                 "this looks like a mock record, do NOT merge it")
    print(f"OK  identity      {got}")
    print(f"OK  prompt_version {r['prompt_version']}")
    print(f"    success={r['success']}  failure_tag={r['failure_tag']}")
    print(f"    prompt_tokens={r['prompt_tokens']}  latency_ms={r['latency_ms']}")
    print(f"    memory_read_tokens={r['memory_read_tokens']} (DP1 budget 1200)")
    print("\nSafe to merge.")


def after() -> None:
    recs = load(MERGED)
    problems = []
    if len(recs) != 1800:
        problems.append(f"record count is {len(recs)}, expected 1800")
    cells = collections.Counter((r["condition"], r["seed"]) for r in recs if r["mode"] == "A")
    bad = {k: v for k, v in cells.items() if v != 120}
    if bad:
        problems.append(f"cells not at 120: {bad}")
    keys = {(r["condition"], r["item_id"], r["seed"], r["mode"]) for r in recs}
    if len(keys) != len(recs):
        problems.append(f"{len(recs) - len(keys)} duplicate keys")
    if not any(r["item_id"] == ITEM and r["condition"] == "b" and r["seed"] == 1 for r in recs):
        problems.append(f"{ITEM} still missing from b/seed 1")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        sys.exit(1)

    b = [r for r in recs if r["condition"] == "b" and r["mode"] == "A"]
    hits = sum(1 for r in b if r["success"])
    print(f"OK  1800/1800 records, all 15 cells at 120, no duplicate keys")
    print(f"OK  {ITEM} present in b/seed 1")
    print(f"\nBaseline Mode A success rate is now {hits}/{len(b)} = {hits / len(b):.4f}")
    print("(was 14/359 = 0.0390 — update Table 4.2 and the audit report to match)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "before":
        before()
    elif mode == "after":
        after()
    else:
        sys.exit(__doc__)
