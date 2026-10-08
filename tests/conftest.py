import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agentmem.llm.client import MockLLM  # noqa: E402
from agentmem.memory import CONDITIONS, make_memory  # noqa: E402
from agentmem.memory.base import Turn  # noqa: E402


@pytest.fixture
def llm():
    return MockLLM()


@pytest.fixture
def turns():
    """Long fact-laden stream: 60 turns, facts + distractors."""
    out = []
    for i in range(60):
        if i % 3 == 0:
            content = f"Update: project P{i} budget is set to £{100 + i}k."
        else:
            content = "The all-hands meeting moved to the main auditorium."
        out.append(Turn(role="system", content=content, step=i))
    return out


def all_memories(llm):
    return {c: make_memory(c, {}, llm=llm) for c in CONDITIONS}
