"""DP1: read() NEVER exceeds budget_tokens, in any condition, at any budget."""

import pytest
from conftest import all_memories

from agentmem.utils.tokens import count_tokens


@pytest.mark.parametrize("budget", [10, 50, 200, 1200])
def test_read_respects_budget(llm, turns, budget):
    for cond, mem in all_memories(llm).items():
        mem.reset()
        for t in turns:
            mem.write(t)
        ctx = mem.read("What is the budget of project P0?", budget_tokens=budget)
        assert ctx.tokens <= budget, f"{cond}: {ctx.tokens} > {budget}"
        # Reported tokens must be consistent with the returned text.
        assert count_tokens(ctx.text) <= budget + 5, cond


def test_zero_budget_returns_empty(llm, turns):
    for cond, mem in all_memories(llm).items():
        mem.reset()
        for t in turns[:10]:
            mem.write(t)
        ctx = mem.read("anything?", budget_tokens=0)
        assert ctx.text == "", cond
        assert ctx.tokens == 0, cond
