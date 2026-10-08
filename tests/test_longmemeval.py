"""Loader + Mode-A regression tests against a fixture in the OFFICIAL
LongMemEval schema (tests/fixtures/longmemeval_fixture.json, 12 items,
2 per question type). Guards the real-data path before the download lands.
"""

from pathlib import Path

from conftest import ROOT

from agentmem.harness.runner import run_mode_a
from agentmem.logging.trajectory import TrajectoryWriter, load_records
from agentmem.tasks.longmemeval import load_longmemeval

FIXTURE = Path(ROOT) / "tests" / "fixtures" / "longmemeval_fixture.json"


def test_loader_parses_official_schema():
    items = load_longmemeval(FIXTURE, seed=1)
    assert len(items) == 12
    for it in items:
        assert it.question and it.answer and it.turns
        assert it.evidence_steps, f"{it.item_id}: no evidence steps"
        assert all(0 <= s < len(it.turns) for s in it.evidence_steps)
        # Evidence turns really contain has_answer content from answer sessions.
        assert all("[" in it.turns[s].content for s in it.evidence_steps)


def test_stratified_cap_preserves_type_mix():
    items = load_longmemeval(FIXTURE, cap=6, seed=1)
    types = [i.question_type for i in items]
    assert len(items) == 6
    assert len(set(types)) == 6, "cap must keep one of each type (stratified)"
    # Deterministic for a fixed seed.
    again = load_longmemeval(FIXTURE, cap=6, seed=1)
    assert [i.item_id for i in items] == [i.item_id for i in again]


def test_mode_a_end_to_end_on_fixture(llm, tmp_path):
    items = load_longmemeval(FIXTURE, cap=4, seed=1)
    writer = TrajectoryWriter(tmp_path / "out.jsonl")
    cfg = {"budgets": {"memory_read_tokens": 300}}
    n = run_mode_a(items, "m1", llm, cfg, writer, seed=1, condition_params={})
    assert n == 4
    recs = load_records(tmp_path / "out.jsonl")
    assert all(r["mode"] == "A" and r["condition"] == "m1" for r in recs)
    assert all(r["memory_read_tokens"] <= 300 for r in recs)
    # Resumability: rerun writes nothing new.
    writer2 = TrajectoryWriter(tmp_path / "out.jsonl")
    assert run_mode_a(items, "m1", llm, cfg, writer2, seed=1, condition_params={}) == 0
