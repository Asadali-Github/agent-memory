"""The memory abstraction — the ONLY thing that varies between conditions.

Locked interface from blueprint §6.3. `read()`'s `budget_tokens` parameter IS
the fairness mechanism (DP1 Equal-Budget Principle); provenance powers the
recall-accuracy metric for free.

Do not implement subclasses until the design is signed off (18 Jun meeting).
Implementations live in: none.py, episodic.py, summary.py, semantic.py, decay.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Turn:
    """One step/observation in an episode (schema finalised in W2)."""
    role: str
    content: str
    step: int
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryContext:
    """What read() returns: text fitted to budget + provenance."""
    text: str
    tokens: int
    provenance: list  # which items were injected, with scores


@dataclass
class MemoryStats:
    tokens_written: int = 0
    tokens_read: int = 0
    items: int = 0
    overhead_calls: int = 0
    latency_ms: float = 0.0


class Memory(ABC):
    @abstractmethod
    def write(self, turn: Turn) -> None:
        """Store a step/observation."""

    @abstractmethod
    def read(self, query: str, budget_tokens: int) -> MemoryContext:
        """Return text fitted to budget + provenance (which items, scores)."""

    @abstractmethod
    def reset(self) -> None:
        """Fresh state per item/episode."""

    @abstractmethod
    def stats(self) -> MemoryStats:
        """tokens_written/read, items, overhead_calls, latency_ms."""
