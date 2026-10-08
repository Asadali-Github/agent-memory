#!/usr/bin/env python
"""Bridge: pipeline run records -> the EDA notebook's live-telemetry format.

The EDA notebook (What_Should_an_Agent_Remember_EDA.ipynb) ingests live data
via `DATA_SOURCE = "live"` + `LIVE_EXPORT_PATH` and gates it with
validate_schema() against RESEARCH_SCHEMA. This script converts
results/runs/*.jsonl into that exact shape, so Stage-2 migration really is the
advertised one-switch change (notebook Section 9.4 checklist).

Field mapping (pipeline -> notebook):
  condition          -> architecture   b=Baseline m1=Episodic m2=Hierarchical
                                       m3=Semantic m4=Decay
  task               -> task_type      longmemeval=LongMemEval,
                                       custom_ds=CustomLongHorizon
  success            -> task_success   (0/1)
  evidence_recall    -> recall_accuracy
  latency_ms         -> latency        (seconds)
  prompt_tokens      -> input_tokens
  completion_tokens  -> output_tokens
  n_retrieved        -> retrieval_depth
  memory_items       -> turn_count
  memory_written_tokens -> horizon_length  (history length in tokens — the
                                            honest live analogue of context_len)
  question_type / evidence_distance -> difficulty (LME: type; custom: terciles)
  failure_tag        -> status ("completed" | failure tag) + error_count

NOTE for the write-up: the notebook's synthetic ARCH_ORDER
(Baseline/Episodic/Semantic/Hierarchical/Retrieval) predates the final design.
For the live run, update ARCH_ORDER in the notebook to the labels emitted here
(Baseline/Episodic/Hierarchical/Semantic/Decay) — extending the canonical
label set for live arms is step 4 of the migration checklist done correctly,
not a relabelling of planted data.

Usage:
  python scripts/export_for_eda.py                       # all runs -> traces.jsonl
  python scripts/export_for_eda.py --out ../traces.jsonl # write next to notebook
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentmem.logging.trajectory import load_records  # noqa: E402

ARCH_LABELS = {"b": "Baseline", "m1": "Episodic", "m2": "Hierarchical",
               "m3": "Semantic", "m4": "Decay"}
TASK_LABELS = {"longmemeval": "LongMemEval", "custom_ds": "CustomLongHorizon"}


def _difficulty(rec: dict) -> str:
    if rec.get("question_type"):          # Mode A: ability type IS the difficulty axis
        return str(rec["question_type"])
    d = rec.get("evidence_distance")      # Mode B: how far back the evidence lies
    if d is None:
        return "medium"
    return "easy" if d < 30 else ("medium" if d < 70 else "hard")


def convert(rec: dict) -> dict:
    return {
        # RESEARCH_SCHEMA columns
        "architecture": ARCH_LABELS.get(rec["condition"], rec["condition"]),
        "task_type": TASK_LABELS.get(rec["task"], rec["task"]),
        "difficulty": _difficulty(rec),
        "horizon_length": rec.get("memory_written_tokens"),
        "turn_count": rec.get("memory_items"),
        "task_success": int(bool(rec["success"])),
        "recall_accuracy": rec.get("evidence_recall"),
        "latency": (rec.get("latency_ms") or 0.0) / 1000.0,
        "input_tokens": rec.get("prompt_tokens"),
        "output_tokens": rec.get("completion_tokens"),
        "retrieval_depth": rec.get("n_retrieved"),
        "error_count": 0 if rec["success"] else 1,
        "status": "completed",
        # provenance extras (harmless to the notebook, vital for the appendix)
        "run_id": rec.get("run_id"), "mode": rec.get("mode"),
        "condition": rec.get("condition"), "item_id": rec.get("item_id"),
        "seed": rec.get("seed"), "failure_tag": rec.get("failure_tag"),
        "overhead_calls": rec.get("overhead_calls"),
        "memory_read_tokens": rec.get("memory_read_tokens"),
        "gbp_spent": rec.get("gbp_spent"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "traces.jsonl"))
    args = ap.parse_args()

    records: list[dict] = []
    for f in sorted(Path(args.runs_dir).glob("*.jsonl")):
        records.extend(load_records(f))
    if not records:
        sys.exit(f"No run records found in {args.runs_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows -> {out}")
    print('Notebook: set DATA_SOURCE = "live", LIVE_EXPORT_PATH = '
          f'"{out.name}" (place file beside the notebook or use a full path).')


if __name__ == "__main__":
    main()
