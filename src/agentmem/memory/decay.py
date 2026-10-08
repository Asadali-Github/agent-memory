"""M4 (Decay): M1 retrieval with the Generative-Agents scoring formula
    score = relevance * exp(-lambda * age)
where age is measured in steps since the turn was written. Isolates the value
of recency-weighting ON TOP of pure relevance (M1 is the ablation partner).
"""

from __future__ import annotations

import math
import time

import numpy as np

from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn
from agentmem.memory.retrieval import (
    fit_to_budget,
    make_scorer,
    reorder_chronologically,
)
from agentmem.utils.tokens import count_tokens


class DecayMemory(Memory):
    def __init__(self, retriever: str = "tfidf", decay_lambda: float = 0.01,
                 chronological_order: bool = False, **_kwargs):
        self._scorer = make_scorer(retriever)
        self._lambda = decay_lambda
        # OFF by default (preserves the reported M4 results). NOTE: with this
        # off, equally-relevant turns are ranked recency-first, so M4's prompt
        # tends REVERSE-chronological — the same layout that cost M2 11.9pp.
        # The M4 deficit is therefore ordering-confounded; see §4.6.
        self._chronological = bool(chronological_order)
        self._turns: list[Turn] = []
        self._clock = 0  # current step count (advances on write)
        self._stats = MemoryStats()

    def write(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._clock = max(self._clock, turn.step)
        self._stats.items += 1
        self._stats.tokens_written += count_tokens(turn.content)

    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        t0 = time.perf_counter()
        rel = self._scorer.score(query, [t.content for t in self._turns])
        combined = np.array([
            float(r) * math.exp(-self._lambda * max(0, self._clock - t.step))
            for r, t in zip(rel, self._turns)
        ])
        order = np.argsort(-combined)
        items = [
            (self._turns[i].content,
             {"step": self._turns[i].step, "role": self._turns[i].role,
              "kind": "raw_turn", "relevance": round(float(rel[i]), 4)},
             float(combined[i]))
            for i in order
        ]
        ctx = fit_to_budget(items, budget_tokens)
        if self._chronological:
            ctx = reorder_chronologically(ctx)
        self._stats.tokens_read += ctx.tokens
        self._stats.latency_ms += (time.perf_counter() - t0) * 1000
        return ctx

    def reset(self) -> None:
        self._turns = []
        self._clock = 0
        self._stats = MemoryStats()

    def stats(self) -> MemoryStats:
        return self._stats
