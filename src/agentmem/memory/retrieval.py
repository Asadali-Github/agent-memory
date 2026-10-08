"""Shared retrieval machinery for M1/M3/M4.

Default scorer is TF-IDF cosine (numpy only — runs on the CPU laptop with zero
model downloads). An optional sentence-transformers backend can be enabled via
config `retriever: embedding` once on HPC; both expose .score(query, texts).

Also home of fit_to_budget(), the single DP1 enforcement point for
retrieval-style memories: items are admitted in score order until the token
budget is exhausted. Every memory read() funnels through budget fitting.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from agentmem.memory.base import MemoryContext
from agentmem.utils.tokens import count_tokens

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfScorer:
    """Cosine similarity over TF-IDF vectors; rebuilt lazily on demand."""

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        docs = [Counter(_terms(t)) for t in texts]
        q = Counter(_terms(query))
        vocab = set(q)
        for d in docs:
            vocab |= set(d)
        vocab_list = sorted(vocab)
        n = len(texts)
        df = Counter()
        for d in docs:
            df.update(set(d))
        idf = {t: math.log((1 + n) / (1 + df[t])) + 1 for t in vocab_list}

        def vec(c: Counter) -> np.ndarray:
            v = np.array([c.get(t, 0) * idf[t] for t in vocab_list], dtype=float)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        qv = vec(q)
        return np.array([float(vec(d) @ qv) for d in docs])


class EmbeddingScorer:
    """sentence-transformers backend (HPC / optional). Lazy model load."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import

        self._model = SentenceTransformer(model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        embs = self._model.encode([query] + texts, normalize_embeddings=True)
        return embs[1:] @ embs[0]


def make_scorer(name: str = "tfidf"):
    if name == "embedding":
        return EmbeddingScorer()
    return TfidfScorer()


def fit_to_budget(
    items: list[tuple[str, dict, float]], budget_tokens: int, sep: str = "\n"
) -> MemoryContext:
    """Admit (text, meta, score) items in given order until budget is spent.

    THE DP1 enforcement point: no memory may return more than budget_tokens.
    """
    chosen: list[str] = []
    provenance: list[dict] = []
    used = 0
    for text, meta, score in items:
        t = count_tokens(text) + (count_tokens(sep) if chosen else 0)
        if used + t > budget_tokens:
            continue  # try smaller later items rather than stopping dead
        chosen.append(text)
        used += t
        provenance.append({**meta, "score": round(float(score), 4)})
    return MemoryContext(text=sep.join(chosen), tokens=used, provenance=provenance)


def reorder_chronologically(ctx: MemoryContext, sep: str = "\n") -> MemoryContext:
    """Re-lay a fitted context oldest-first WITHOUT changing what was selected.

    Selection stays score-first (so the highest-scoring turns still win the
    budget) and only the prompt layout changes.

    WHY THIS EXISTS. Retrieval-style memories (M1, M4) emit turns in score
    order, so their prompts are neither chronological nor consistently ordered,
    while B and M2 emit chronologically. That is the same class of confound that
    made M2 look like a harmful architecture (see B1 in the expert review):
    a contrast between a score-ordered and a chronologically-ordered condition
    mixes architecture with presentation. M4 is the sharpest case — among
    equally relevant turns its decay term ranks by recency, producing a
    reverse-chronological prompt, exactly the layout that cost M2 11.9pp.

    Left OFF by default so existing results stay valid. Turn it on per condition
    via `chronological_order: true` and pair it with the b_reversed control arm.
    """
    steps = [p.get("step") for p in ctx.provenance]
    if any(s is None for s in steps) or steps == sorted(steps):
        return ctx
    order = sorted(range(len(ctx.provenance)), key=lambda i: ctx.provenance[i]["step"])
    parts = ctx.text.split(sep)
    if len(parts) != len(ctx.provenance):
        return ctx  # summary blocks or separators inside text: leave alone
    return MemoryContext(
        text=sep.join(parts[i] for i in order),
        tokens=ctx.tokens,
        provenance=[ctx.provenance[i] for i in order],
    )
