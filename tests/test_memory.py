"""Interface conformance + architecture-specific behaviour for every condition."""

from conftest import all_memories

from agentmem.memory.base import MemoryContext, MemoryStats, Turn


def test_interface_conformance(llm, turns):
    for cond, mem in all_memories(llm).items():
        mem.reset()
        for t in turns[:12]:
            mem.write(t)
        ctx = mem.read("What is the budget of project P0?", budget_tokens=500)
        assert isinstance(ctx, MemoryContext), cond
        assert isinstance(ctx.provenance, list), cond
        st = mem.stats()
        assert isinstance(st, MemoryStats), cond
        assert st.items == 12, cond
        assert st.tokens_written > 0, cond
        mem.reset()
        assert mem.stats().items == 0, cond
        assert mem.read("q?", 100).text == "", f"{cond}: reset must clear state"


def test_baseline_keeps_most_recent(llm):
    from agentmem.memory.none import NoMemory

    mem = NoMemory()
    for i in range(50):
        mem.write(Turn(role="user", content=f"message number {i} with padding words",
                       step=i))
    ctx = mem.read("irrelevant?", budget_tokens=60)
    steps = [p["step"] for p in ctx.provenance]
    assert steps, "baseline returned nothing"
    assert min(steps) > 30, f"recency window should hold recent turns, got {steps}"


def test_episodic_retrieves_relevant_over_recent(llm):
    from agentmem.memory.episodic import EpisodicMemory

    mem = EpisodicMemory()
    mem.write(Turn(role="system",
                   content="Update: project Falcon budget is set to £250k.", step=0))
    for i in range(1, 40):
        mem.write(Turn(role="system",
                       content="The all-hands meeting moved to the auditorium.", step=i))
    ctx = mem.read("What is the budget of project Falcon?", budget_tokens=60)
    assert any(p["step"] == 0 for p in ctx.provenance), \
        "episodic memory should retrieve the old relevant turn"


def test_summary_folds_and_charges_overhead(llm):
    from agentmem.memory.summary import SummaryMemory

    mem = SummaryMemory(llm=llm, buffer_token_threshold=100)
    for i in range(30):
        mem.write(Turn(role="system",
                       content=f"Update: project Nimbus budget is set to £{i}k.", step=i))
    assert mem.stats().overhead_calls > 0, "summariser never ran"
    ctx = mem.read("What is the budget of project Nimbus?", budget_tokens=400)
    assert any(p["kind"] == "summary" for p in ctx.provenance)


def test_semantic_update_on_conflict(llm):
    from agentmem.memory.semantic import SemanticMemory

    mem = SemanticMemory(llm=llm, extract_every=1)
    mem.write(Turn(role="system",
                   content="Update: project Zephyr budget is set to £100k.", step=0))
    mem.write(Turn(role="system",
                   content="Update: project Zephyr budget is set to £999k.", step=1))
    ctx = mem.read("What is the budget of project Zephyr?", budget_tokens=300)
    assert "999" in ctx.text, "newest value must win"
    assert "100k" not in ctx.text, "stale value should have been replaced"


def test_decay_prefers_recent_among_equally_relevant(llm):
    from agentmem.memory.decay import DecayMemory

    mem = DecayMemory(decay_lambda=0.1)
    mem.write(Turn(role="system",
                   content="Update: project Vortex status is blocked.", step=0))
    mem.write(Turn(role="system",
                   content="Update: project Vortex status is on track.", step=50))
    ctx = mem.read("What is the status of project Vortex?", budget_tokens=20)
    assert ctx.provenance and ctx.provenance[0]["step"] == 50, \
        "decay should rank the recent duplicate first"


def test_declared_prompt_ordering_policy(llm):
    """Pin each condition's prompt ORDER, because ordering is a live confound.

    B and M2 lay turns out chronologically. M1 and M4 lay them out in score
    order by default — which is why any B-vs-M1 or B-vs-M4 contrast currently
    mixes architecture with presentation, the same way the pre-patch M2
    contrast did. This test exists so that if anyone changes an ordering policy,
    they are forced to notice that the affected results need regenerating.
    """
    from agentmem.memory import make_memory

    chronological = {"b": True, "m2": True, "m1": False, "m3": False, "m4": False}
    for cond, expect_chrono in chronological.items():
        mem = make_memory(cond, {}, llm=llm)
        for i in range(30):
            mem.write(Turn(role="system",
                           content=f"Update: project Nimbus milestone {i} logged.",
                           step=i))
        ctx = mem.read("What milestone was logged?", budget_tokens=300)
        steps = [p["step"] for p in ctx.provenance if p.get("kind") != "summary"]
        if expect_chrono:
            assert steps == sorted(steps), \
                f"{cond} must emit chronologically, got {steps}"


def test_retrieval_memories_can_opt_into_chronological_order(llm):
    """M1/M4 must be able to produce chronological prompts on request.

    This is what makes the ordering confound testable: run M1/M4 twice, once
    each way, and the difference is attributable to layout alone.
    """
    from agentmem.memory import make_memory

    for cond in ("m1", "m4"):
        mem = make_memory(cond, {"chronological_order": True}, llm=llm)
        for i in range(30):
            mem.write(Turn(role="system",
                           content=f"Update: project Nimbus milestone {i} logged.",
                           step=i))
        ctx = mem.read("What milestone was logged?", budget_tokens=300)
        steps = [p["step"] for p in ctx.provenance]
        assert steps == sorted(steps), \
            f"{cond} with chronological_order=True must emit oldest-first, got {steps}"
        assert ctx.text.split("\n")[0].endswith(f"milestone {steps[0]} logged."), \
            f"{cond}: text and provenance must be reordered together"


def test_m3_chrono_is_a_layout_only_control(llm):
    """m3_chrono must differ from m3 in prompt ORDER and nothing else.

    The arm was originally run via an HPC-side patch (job 2848922); this test
    pins the restored in-repo implementation: same fact store, same selection,
    same token count — only the emitted layout changes (oldest-first by
    earliest source step).
    """
    from agentmem.memory import make_memory

    def run(cond):
        mem = make_memory(cond, {"chronological_order": cond.endswith("chrono")},
                          llm=llm)
        for i in range(30):
            mem.write(Turn(role="system",
                           content=f"Update: project Nimbus milestone {i} logged.",
                           step=i))
        return mem.read("What milestone was logged?", budget_tokens=300)

    score_ordered = run("m3")
    chrono = run("m3_chrono")
    assert score_ordered.provenance, "m3 must extract facts under MockLLM"
    assert [p["step"] for p in chrono.provenance] == \
        sorted(p["step"] for p in chrono.provenance), \
        "m3_chrono must emit oldest-first by earliest source step"
    assert {p["steps"][0] for p in chrono.provenance} == \
        {p["steps"][0] for p in score_ordered.provenance}, \
        "m3_chrono must select the SAME facts as m3"
    assert chrono.tokens == score_ordered.tokens, \
        "layout control must not change the token count"


def test_b_reversed_is_ordering_only_control():
    """b_reversed must differ from b in prompt ORDER and nothing else.

    If selection or token count diverged, the arm would no longer isolate the
    ordering effect and could not be subtracted from the M2 contrast.
    """
    from agentmem.memory import make_memory

    turns = [Turn(role="user", content=f"turn {i} " + ("padding " * i), step=i)
             for i in range(40)]
    for budget in (120, 400):
        out = {}
        for cond, params in [("b", {}), ("b_reversed", {"chronological_order": False})]:
            mem = make_memory(cond, params)
            for t in turns:
                mem.write(t)
            out[cond] = mem.read("q?", budget_tokens=budget)

        b, rev = out["b"], out["b_reversed"]
        assert sorted(p["step"] for p in b.provenance) == \
               sorted(p["step"] for p in rev.provenance), \
            f"budget={budget}: control arm must select the same turns as b"
        assert b.tokens == rev.tokens, \
            f"budget={budget}: control arm must spend the same read budget as b"

        b_steps = [p["step"] for p in b.provenance]
        rev_steps = [p["step"] for p in rev.provenance]
        assert len(b_steps) > 1, "need >1 turn for ordering to be observable"
        assert b_steps == sorted(b_steps), "b must be chronological"
        assert rev_steps == sorted(rev_steps, reverse=True), \
            "b_reversed must be reverse-chronological"
        assert b.text != rev.text, "the two arms must differ in prompt layout"


def test_provenance_is_not_truncated_by_the_runner():
    """Regression: runner.py logged provenance[:20] AFTER sorting it.

    Because B sorts ascending and the reverse-chronological arms sort
    descending, truncating to the first 20 entries made the two conditions
    appear to select disjoint turn sets (Jaccard 0.0 on every item) when in
    fact they select identical sets. Any ordering/inversion audit run against
    truncated provenance is invalid. Pin the full-provenance contract.
    """
    import inspect
    from agentmem.harness import runner

    src = inspect.getsource(runner)
    assert 'ctx_info["provenance"][:20]' not in src, (
        "runner must log FULL provenance; truncation silently corrupts the "
        "prompt-order audit (see scripts/audit_prompt_order.py)"
    )


def test_b_reversed_is_an_ordering_only_control():
    """b_reversed must differ from b in LAYOUT ONLY, never in selection."""
    from agentmem.memory.base import Turn
    from agentmem.memory.none import NoMemory

    turns = [Turn(step=i, role="system", content=f"Turn {i}: note number {i} here.")
             for i in range(60)]

    def read(chrono):
        m = NoMemory(chronological_order=chrono)
        for t in turns:
            m.write(t)
        return m.read("what is the budget?", 400)

    a, b = read(True), read(False)
    sa = [p["step"] for p in a.provenance]
    sb = [p["step"] for p in b.provenance]
    assert set(sa) == set(sb), "selection must be identical"
    assert a.tokens == b.tokens, "token count must be identical"
    assert sb == list(reversed(sa)), "layout must be exactly reversed"


def test_all_modules_import_under_python38():
    """Regression: CERES serves vLLM under Python 3.8.

    PEP 585 generics (list[dict]) and PEP 604 unions (dict | None) are
    evaluated at import time on 3.8 and raise TypeError unless the module
    opts into lazy annotations. A code upload once clobbered a hand-applied
    cluster-side fix and every grid cell died on import. Pin the contract.
    """
    import os
    import re

    root = os.path.join(os.path.dirname(__file__), "..")
    offenders = []
    for top in ("src", "scripts", "analysis"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, top)):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".pytest_cache")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                if "from __future__ import annotations" in src:
                    continue
                if re.search(r"->\s*(list|dict|set|tuple|type)\[|"
                             r":\s*(list|dict|set|tuple)\[|\|\s*None", src):
                    offenders.append(os.path.relpath(path, root))
    assert not offenders, (
        "these modules use 3.9+ annotation syntax without "
        "`from __future__ import annotations` and will fail on CERES: "
        + ", ".join(sorted(offenders))
    )
