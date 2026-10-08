"""B (NoMemory baseline): sliding window of the MOST RECENT turns fitting the budget.

Built first because it defines fairness: every other condition gets exactly the
same read budget — they just choose differently what to spend it on.
"""

from __future__ import annotations

import time

from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn
from agentmem.memory.retrieval import fit_to_budget
from agentmem.utils.tokens import count_tokens


class NoMemory(Memory):
    """Recency-window baseline.

    `chronological_order` exists to support the ordering-only control arm
    (`b_reversed`). Selection is always recency-first so the same turns win the
    budget under both settings; the flag changes ONLY the order in which those
    turns are laid out in the prompt. That isolates prompt ordering from memory
    architecture — see configs/conditions/b_reversed.yaml.
    """

    def __init__(self, chronological_order: bool = True, **_kwargs):
        self._turns: list[Turn] = []
        self._stats = MemoryStats()
        self._chronological = bool(chronological_order)

    def write(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._stats.items += 1
        self._stats.tokens_written += count_tokens(turn.content)

    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        t0 = time.perf_counter()
        # Most recent first so recency wins the budget; restore chronology after.
        recent = list(reversed(self._turns))
        items = [
            (t.content, {"step": t.step, "role": t.role, "kind": "raw_turn"}, 1.0)
            for t in recent
        ]
        ctx = fit_to_budget(items, budget_tokens)
        # Re-order chosen text chronologically for the prompt (unless the
        # ordering-control arm has asked for the reverse-chronological layout).
        chosen_steps = {p["step"] for p in ctx.provenance}
        source = self._turns if self._chronological else recent
        ordered = [t.content for t in source if t.step in chosen_steps]
        provenance = sorted(ctx.provenance, key=lambda p: p["step"],
                            reverse=not self._chronological)
        ctx = MemoryContext(text="\n".join(ordered), tokens=ctx.tokens,
                            provenance=provenance)
        self._stats.tokens_read += ctx.tokens
        self._stats.latency_ms += (time.perf_counter() - t0) * 1000
        return ctx

    def reset(self) -> None:
        self._turns = []
        self._stats = MemoryStats()

    def stats(self) -> MemoryStats:
        return self._stats
