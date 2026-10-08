"""M3 (Structured semantic): the fixed LLM extracts atomic facts from turns into
a fact store (update-on-conflict, Mem0-style but custom — auditable and free).
read() retrieves the most query-relevant FACTS, not raw turns.

Extraction is batched every `extract_every` turns; each batch = 1 overhead call.
Conflict policy: a new fact whose key (first ~6 content words) matches an
existing fact REPLACES it (newest-wins), mirroring Mem0's update operation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from agentmem.llm.client import LLMClient
from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn
from agentmem.memory.retrieval import (
    fit_to_budget,
    make_scorer,
    reorder_chronologically,
)
from agentmem.utils.tokens import count_tokens, truncate_to_tokens

EXTRACT_PROMPT = (
    "Extract the atomic facts from this conversation excerpt. One fact per "
    "line, each starting with '- '. Keep names, numbers, dates, decisions and "
    "preferences exactly. Skip chit-chat.\n\nEXCERPT:\n{chunk}\n\nFACTS:"
)

_STOP = {"the", "a", "an", "is", "was", "are", "were", "to", "of", "and", "in",
         "on", "for", "that", "this", "user", "assistant"}


@dataclass
class Fact:
    text: str
    key: str
    source_steps: list[int]


def _fact_key(text: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
             if w not in _STOP and not any(ch.isdigit() for ch in w)]
    return " ".join(words[:6])


class SemanticMemory(Memory):
    def __init__(
        self,
        llm: LLMClient | None = None,
        retriever: str = "tfidf",
        extract_every: int = 5,
        chronological_order: bool = False,
        **_kwargs,
    ):
        if llm is None:
            raise ValueError("SemanticMemory needs the shared fixed LLM client")
        self._llm = llm
        self._scorer = make_scorer(retriever)
        self._extract_every = extract_every
        # OFF by default: the reported M3 results were produced score-ordered.
        # m3_chrono turns this on: facts are re-laid oldest-first (by earliest
        # source step) after selection, an ordering-only control for M3. Note
        # this repairs LAYOUT only — extraction already merged conflicts at
        # write time, which is why the m3_chrono arm did not recover (0.740
        # vs 0.753; see supervision/ADDENDUM_m3_chrono_2026-07-30.md).
        self._chronological = bool(chronological_order)
        self._facts: list[Fact] = []
        self._pending: list[Turn] = []
        self._stats = MemoryStats()

    def write(self, turn: Turn) -> None:
        self._pending.append(turn)
        self._stats.items += 1
        self._stats.tokens_written += count_tokens(turn.content)
        if len(self._pending) >= self._extract_every:
            self._extract()

    # Largest excerpt sent to the extractor in ONE call. Mode A (LongMemEval)
    # sessions can be thousands of tokens, so extraction is split into
    # serving-window-sized pieces; each piece is its own overhead call.
    _MAX_EXTRACT_CHUNK_TOKENS = 900

    def _extract(self) -> None:
        if not self._pending:
            return
        steps = [t.step for t in self._pending]
        lines_in = [f"[{t.role} step {t.step}] {t.content}" for t in self._pending]
        self._pending = []

        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for line_in in lines_in:
            line_tokens = count_tokens(line_in)
            if line_tokens > self._MAX_EXTRACT_CHUNK_TOKENS:
                if current:
                    pieces.append("\n".join(current)); current, current_tokens = [], 0
                pieces.append(truncate_to_tokens(line_in, self._MAX_EXTRACT_CHUNK_TOKENS))
                continue
            if current_tokens + line_tokens > self._MAX_EXTRACT_CHUNK_TOKENS:
                pieces.append("\n".join(current)); current, current_tokens = [], 0
            current.append(line_in); current_tokens += line_tokens
        if current:
            pieces.append("\n".join(current))

        response_lines: list[str] = []
        for piece in pieces:
            resp = self._llm.generate(
                [{"role": "user", "content": EXTRACT_PROMPT.format(chunk=piece)}],
                max_tokens=400,
            )
            self._stats.overhead_calls += 1
            response_lines.extend(resp.text.splitlines())

        for line in response_lines:
            line = line.strip()
            if not line.startswith("- ") or len(line) < 4:
                continue
            text = line[2:].strip()
            key = _fact_key(text)
            if not key:
                continue
            # Update-on-conflict: newest fact with the same key wins.
            existing = next((f for f in self._facts if f.key == key), None)
            if existing:
                existing.text = text
                existing.source_steps = sorted(set(existing.source_steps + steps))
            else:
                self._facts.append(Fact(text=text, key=key, source_steps=steps))

    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        t0 = time.perf_counter()
        self._extract()  # flush pending turns so nothing recent is invisible
        scores = self._scorer.score(query, [f.text for f in self._facts])
        order = np.argsort(-scores) if len(scores) else []
        items = [
            (self._facts[i].text,
             {"kind": "fact", "steps": self._facts[i].source_steps,
              "step": self._facts[i].source_steps[0] if self._facts[i].source_steps else -1},
             float(scores[i]))
            for i in order
        ]
        ctx = fit_to_budget(items, budget_tokens)
        if self._chronological:
            ctx = reorder_chronologically(ctx)
        self._stats.tokens_read += ctx.tokens
        self._stats.latency_ms += (time.perf_counter() - t0) * 1000
        return ctx

    def reset(self) -> None:
        self._facts = []
        self._pending = []
        self._stats = MemoryStats()

    def stats(self) -> MemoryStats:
        return self._stats
