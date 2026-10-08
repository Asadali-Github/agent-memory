"""Memory package: locked ABC + the five conditions + factory."""

# CERES serves vLLM under Python 3.8, where PEP 585 generics (list[dict]) and
# PEP 604 unions (dict | None) raise TypeError at import time. This makes all
# annotations lazy strings so the module imports on 3.8. Do not remove.
from __future__ import annotations

from agentmem.memory.base import Memory, MemoryContext, MemoryStats, Turn

CONDITIONS = ("b", "m1", "m2", "m3", "m4")

# Control arms: run on demand (--condition b_reversed), NOT part of the main
# grid, so --all-conditions and the registered analysis stay as pre-specified.
CONTROL_ARMS = ("b_reversed", "m1_chrono", "m3_chrono", "m4_chrono")
ALL_ARMS = CONDITIONS + CONTROL_ARMS


def make_memory(condition: str, config: dict | None = None, llm=None) -> Memory:
    """Factory: condition id -> Memory instance.

    b  = NoMemory (recency window)      m1 = EpisodicMemory (retrieval)
    m2 = SummaryMemory (hierarchical)   m3 = SemanticMemory (fact store)
    m4 = DecayMemory (retrieval+decay)

    b_reversed = NoMemory with the chronological re-sort disabled. Ordering-only
    control: identical turn selection to b, reverse-chronological prompt layout.
    """
    from agentmem.memory.decay import DecayMemory
    from agentmem.memory.episodic import EpisodicMemory
    from agentmem.memory.none import NoMemory
    from agentmem.memory.semantic import SemanticMemory
    from agentmem.memory.summary import SummaryMemory

    cfg = dict(config or {})
    condition = condition.lower()
    if condition == "b":
        return NoMemory(**cfg)
    if condition == "b_reversed":
        cfg.setdefault("chronological_order", False)
        return NoMemory(**cfg)
    if condition == "m1":
        return EpisodicMemory(**cfg)
    if condition == "m1_chrono":
        cfg.setdefault("chronological_order", True)
        return EpisodicMemory(**cfg)
    if condition == "m2":
        return SummaryMemory(llm=llm, **cfg)
    if condition == "m3":
        return SemanticMemory(llm=llm, **cfg)
    if condition == "m3_chrono":
        cfg.setdefault("chronological_order", True)
        return SemanticMemory(llm=llm, **cfg)
    if condition == "m4":
        return DecayMemory(**cfg)
    if condition == "m4_chrono":
        cfg.setdefault("chronological_order", True)
        return DecayMemory(**cfg)
    raise ValueError(f"Unknown condition: {condition!r} (expected one of {ALL_ARMS})")


__all__ = ["Memory", "MemoryContext", "MemoryStats", "Turn", "make_memory",
           "CONDITIONS", "CONTROL_ARMS", "ALL_ARMS"]
