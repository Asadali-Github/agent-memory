"""Metrics/eval are pure functions over logged data — test them as such."""

import math

from agentmem.eval.failures import tag_failure
from agentmem.eval.judge import exact_match
from agentmem.eval.recall import evidence_metrics
from agentmem.tasks.custom_ds import generate_episode, grade_answer
from agentmem.utils.costs import BudgetExceededError, CostTracker
from agentmem.utils.tokens import count_tokens, truncate_to_tokens


def test_evidence_metrics():
    prov = [{"step": 1}, {"step": 5}, {"kind": "fact", "steps": [7, 9]}]
    m = evidence_metrics(prov, evidence_steps=[5, 7, 100])
    assert math.isclose(m["evidence_recall"], 2 / 3)
    assert math.isclose(m["evidence_precision"], 2 / 4)
    assert evidence_metrics(prov, [])["evidence_recall"] is None


def test_exact_match_normalisation():
    assert exact_match("The deadline is 14 July 2026.", "14 July 2026")
    assert exact_match("£250k", "250k")
    assert not exact_match("I don't know.", "14 July 2026")


def test_failure_tags():
    assert tag_failure(True, "x", 1.0) == "success"
    assert tag_failure(False, "", 1.0) == "no_answer"
    assert tag_failure(False, "wrong", 0.0) == "retrieval_miss"
    assert tag_failure(False, "wrong", 0.5) == "partial_retrieval"
    assert tag_failure(False, "wrong", 1.0) == "reasoning_failure"
    assert tag_failure(False, "old value", 1.0, stale_hit=True) == "stale_fact"


def test_custom_task_deterministic_and_answerable():
    ep1 = generate_episode("e", seed=42)
    ep2 = generate_episode("e", seed=42)
    assert [t.content for t in ep1.turns] == [t.content for t in ep2.turns]
    assert ep1.queries, "episode must contain queries"
    for q in ep1.queries:
        assert q.evidence_steps and max(q.evidence_steps) < q.step
        # Gold value really is stated at the evidence step.
        assert any(q.answer in ep1.turns[s].content for s in q.evidence_steps)
    assert grade_answer(f"The answer is {ep1.queries[0].answer}.",
                        ep1.queries[0].answer)


def test_tokens_and_truncate():
    text = "word " * 400
    assert count_tokens(text) > 100
    cut = truncate_to_tokens(text, 50)
    assert count_tokens(cut) <= 50


def test_cost_kill_switch():
    t = CostTracker(price_key="groq/llama-3.3-70b-versatile", gbp_cap=0.0001)
    try:
        t.add(1_000_000, 1_000_000)
        raise AssertionError("kill-switch did not fire")
    except BudgetExceededError:
        pass
