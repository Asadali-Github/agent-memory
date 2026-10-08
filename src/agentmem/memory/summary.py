"""M2 (Summary/hierarchical, MemGPT-style): running summary maintained by the
SAME fixed LLM; oldest raw turns are folded into the summary when the raw
buffer exceeds a threshold. Summarisation calls are charged to the condition
as overhead (overhead_calls + the tracker inside the shared LLM client).

read() returns: [running summary] + [most recent raw turns], fitted to budget.
Query-independent by design — that is the architectural contrast with M1.
"""

from __future__ import annotations

import time

from agentmem.llm.client import LLMClient
from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn
from agentmem.memory.retrieval import fit_to_budget
from agentmem.utils.tokens import count_tokens, truncate_to_tokens

SUMMARISE_PROMPT = (
    "Summarize the following conversation excerpt into concise notes, "
    "PRESERVING all concrete facts (names, numbers, dates, decisions, "
    "preferences). Merge with the existing notes.\n\n"
    "EXISTING NOTES:\n{summary}\n\nNEW EXCERPT:\n{chunk}\n\nUPDATED NOTES:"
)


class SummaryMemory(Memory):
    def __init__(
        self,
        llm: LLMClient | None = None,
        buffer_token_threshold: int = 1500,
        fold_fraction: float = 0.5,
        **_kwargs,
    ):
        if llm is None:
            raise ValueError("SummaryMemory needs the shared fixed LLM client")
        self._llm = llm
        self._threshold = buffer_token_threshold
        self._fold_fraction = fold_fraction
        self._summary = ""
        self._summary_sources: list[int] = []  # steps folded into the summary
        self._buffer: list[Turn] = []
        self._stats = MemoryStats()

    def write(self, turn: Turn) -> None:
        self._buffer.append(turn)
        self._stats.items += 1
        self._stats.tokens_written += count_tokens(turn.content)
        if sum(count_tokens(t.content) for t in self._buffer) > self._threshold:
            self._fold_oldest()

    # Largest excerpt sent to the summariser in ONE call. Mode A (LongMemEval)
    # sessions can be thousands of tokens each, so a single fold must be split
    # into serving-window-sized pieces (MemGPT folds recursively for the same
    # reason). Each piece is charged as its own overhead call, so the cost
    # accounting stays honest.
    _MAX_FOLD_CHUNK_TOKENS = 900

    def _fold_oldest(self) -> None:
        n = max(1, int(len(self._buffer) * self._fold_fraction))
        oldest, self._buffer = self._buffer[:n], self._buffer[n:]

        # Build fold pieces that fit the summariser's input window.
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for t in oldest:
            line = f"[{t.role} step {t.step}] {t.content}"
            line_tokens = count_tokens(line)
            if line_tokens > self._MAX_FOLD_CHUNK_TOKENS:
                # A single oversized turn becomes its own (truncated) piece.
                if current:
                    pieces.append("\n".join(current)); current, current_tokens = [], 0
                pieces.append(truncate_to_tokens(line, self._MAX_FOLD_CHUNK_TOKENS))
                continue
            if current_tokens + line_tokens > self._MAX_FOLD_CHUNK_TOKENS:
                pieces.append("\n".join(current)); current, current_tokens = [], 0
            current.append(line); current_tokens += line_tokens
        if current:
            pieces.append("\n".join(current))

        for piece in pieces:
            # Prior notes are kept compact so (notes + excerpt + 400 output)
            # fits comfortably inside the smallest serving window we use.
            prior = truncate_to_tokens(self._summary or "(none)", 500)
            resp = self._llm.generate(
                [{"role": "user",
                  "content": SUMMARISE_PROMPT.format(summary=prior, chunk=piece)}],
                max_tokens=400,
            )
            self._summary = resp.text.strip()
            self._stats.overhead_calls += 1

        self._summary_sources.extend(t.step for t in oldest)

    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        t0 = time.perf_counter()
        items: list[tuple[str, dict, float]] = []
        if self._summary:
            summary_text = truncate_to_tokens(
                f"[MEMORY NOTES]\n{self._summary}", max(0, budget_tokens - 8))
            if summary_text:
                items.append(
                    (summary_text,
                     {"kind": "summary", "steps": list(self._summary_sources),
                      "step": -1},
                     2.0)
                )
        for t in reversed(self._buffer):  # newest raw turns first
            items.append(
                (t.content, {"step": t.step, "role": t.role, "kind": "raw_turn"}, 1.0)
            )
        ctx = fit_to_budget(items, budget_tokens)
        # Selection is recency-first (above), but the PROMPT must be chronological,
        # exactly as NoMemory.read() does. Without this the only difference between
        # M2 and B on short histories is turn ordering, which confounds the
        # architecture contrast with a presentation artefact.
        chosen = {p["step"] for p in ctx.provenance}
        ordered_parts: list[str] = []
        if -1 in chosen and self._summary:
            ordered_parts.append(
                truncate_to_tokens(f"[MEMORY NOTES]\n{self._summary}",
                                   max(0, budget_tokens - 8))
            )
        ordered_parts += [t.content for t in self._buffer if t.step in chosen]
        ctx = MemoryContext(
            text="\n".join(ordered_parts),
            tokens=ctx.tokens,
            provenance=sorted(ctx.provenance, key=lambda p: p["step"]),
        )
        self._stats.tokens_read += ctx.tokens
        self._stats.latency_ms += (time.perf_counter() - t0) * 1000
        return ctx

    def reset(self) -> None:
        self._summary = ""
        self._summary_sources = []
        self._buffer = []
        self._stats = MemoryStats()

    def stats(self) -> MemoryStats:
        return self._stats
