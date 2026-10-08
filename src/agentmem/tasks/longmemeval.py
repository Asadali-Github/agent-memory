"""LongMemEval loader + stratified capper (Mode A: memory probing).

Data file (not committed; see data/README): LongMemEval_s.json from
https://github.com/xiaowu0162/LongMemEval — a list of items with fields:
  question_id, question_type, question, answer,
  haystack_session_ids, haystack_dates, haystack_sessions (list of sessions,
  each a list of {"role": ..., "content": ...} turns),
  answer_session_ids  (which sessions contain the evidence)

Capping: stratified by question_type (equal share per type, seeded), so the
capped subset preserves the ability mix — decided 14 Jun.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agentmem.memory.base import Turn


@dataclass
class ModeAItem:
    item_id: str
    question_type: str
    question: str
    answer: str
    turns: list[Turn]                    # flattened history to ingest
    evidence_steps: list[int] = field(default_factory=list)  # gold steps
    metadata: dict = field(default_factory=dict)


def _flatten_sessions(item: dict) -> tuple[list[Turn], list[int]]:
    """Flatten haystack sessions to a single turn stream; mark evidence steps."""
    turns: list[Turn] = []
    evidence_steps: list[int] = []
    answer_ids = set(item.get("answer_session_ids", []))
    step = 0
    session_ids = item.get("haystack_session_ids", [])
    dates = item.get("haystack_dates", [])
    for si, session in enumerate(item.get("haystack_sessions", [])):
        sid = session_ids[si] if si < len(session_ids) else f"s{si}"
        date = dates[si] if si < len(dates) else ""
        for turn in session:
            content = f"[{date}] {turn['role']}: {turn['content']}" if date else \
                      f"{turn['role']}: {turn['content']}"
            turns.append(Turn(role=turn["role"], content=content, step=step,
                              metadata={"session_id": sid}))
            # Gold evidence: any turn in an answer session flagged has_answer,
            # else all turns of that session count as evidence.
            if sid in answer_ids and turn.get("has_answer", True):
                evidence_steps.append(step)
            step += 1
    return turns, evidence_steps


def load_longmemeval(
    path: str | Path,
    cap: int | None = None,
    seed: int = 1,
    exclude_types: tuple[str, ...] = (),
) -> list[ModeAItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[ModeAItem] = []
    for it in raw:
        qtype = it.get("question_type", "unknown")
        if qtype in exclude_types:
            continue
        turns, evidence = _flatten_sessions(it)
        items.append(ModeAItem(
            item_id=str(it["question_id"]),
            question_type=qtype,
            question=it["question"],
            answer=str(it.get("answer", "")),
            turns=turns,
            evidence_steps=evidence,
            metadata={"n_turns": len(turns)},
        ))
    if cap is None or cap >= len(items):
        return items
    return _stratified_cap(items, cap, seed)


def _stratified_cap(items: list[ModeAItem], cap: int, seed: int) -> list[ModeAItem]:
    by_type: dict[str, list[ModeAItem]] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)
    rng = random.Random(seed)
    types = sorted(by_type)
    per_type = max(1, cap // len(types))
    chosen: list[ModeAItem] = []
    for t in types:
        pool = sorted(by_type[t], key=lambda x: x.item_id)
        rng.shuffle(pool)
        chosen.extend(pool[:per_type])
    # Top up to exactly `cap` from the leftovers, deterministically.
    if len(chosen) < cap:
        leftovers = [it for t in types for it in by_type[t] if it not in chosen]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: cap - len(chosen)])
    return chosen[:cap]
