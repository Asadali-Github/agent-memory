"""JSONL run records: one line per (item|query) x condition x seed.

RESEARCH_SCHEMA — every record carries:
  identity   : run_id, mode, task, condition, item_id, seed
  outcome    : question, gold, prediction, success, failure_tag
  memory     : evidence_recall, evidence_precision, provenance (truncated)
  cost       : prompt_tokens, completion_tokens, memory_read_tokens,
               memory_written_tokens, overhead_calls, latency_ms, gbp_spent
Resumable: existing (condition, item_id, seed, mode) keys are skipped on rerun.
"""

from __future__ import annotations

import json
from pathlib import Path


def record_key(rec: dict) -> tuple:
    return (rec["condition"], rec["item_id"], rec["seed"], rec["mode"])


class TrajectoryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[tuple] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self._done.add(record_key(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue

    def is_done(self, condition: str, item_id: str, seed: int, mode: str) -> bool:
        return (condition, item_id, seed, mode) in self._done

    def write(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._done.add(record_key(rec))


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
