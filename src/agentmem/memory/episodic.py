"""M1 (Episodic/retrieval): store every turn verbatim; read = top-k similarity
fitted to budget.

Scorer is TF-IDF by default (CPU-safe); switch to sentence-transformer
embeddings with config `retriever: embedding` on HPC. The architecture claim
(selective retrieval beats recency) is scorer-agnostic; report which was used.
"""

from __future__ import annotations

import time

import numpy as np

from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn
from agentmem.memory.retrieval import (
    fit_to_budget,
    make_scorer,
    reorder_chronologically,
)
from agentmem.utils.tokens import count_tokens


class EpisodicMemory(Memory):
    def __init__(self, retriever: str = "tfidf", chronological_order: bool = False,
                 **_kwargs):
        self._scorer = make_scorer(retriever)
        self._retriever_name = retriever
        # OFF by default: retrieval memories have always emitted score-ordered
        # prompts and the reported M1 results were produced that way. See
        # reorder_chronologically() for why you may want to turn this on.
        self._chronological = bool(chronological_order)
        self._turns: list[Turn] = []
        self._stats = MemoryStats()

    def write(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._stats.items += 1
        self._stats.tokens_written += count_tokens(turn.content)

    def _rank(self, query: str) -> list[tuple[Turn, float]]:
        scores = self._scorer.score(query, [t.content for t in self._turns])
        order = np.argsort(-scores)
        return [(self._turns[i], float(scores[i])) for i in order]

    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        t0 = time.perf_counter()
        ranked = self._rank(query)
        items = [
            (t.content, {"step": t.step, "role": t.role, "kind": "raw_turn"}, s)
            for t, s in ranked
        ]
        ctx = fit_to_budget(items, budget_tokens)
        if self._chronological:
            ctx = reorder_chronologically(ctx)
        self._stats.tokens_read += ctx.tokens
        self._stats.latency_ms += (time.perf_counter() - t0) * 1000
        return ctx

    def reset(self) -> None:
        self._turns = []
        self._stats = MemoryStats()

    def stats(self) -> MemoryStats:
        return self._stats
