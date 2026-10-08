"""Run loops for both evaluation modes.

Mode A (memory probing, LongMemEval): ingest the full history into the memory,
then ask the probe question with a budget-limited read.

Mode B (memory use during execution, custom task): stream observations into
memory chronologically; at each query step, read + answer BEFORE continuing.

Both emit one JSONL record per question (RESEARCH_SCHEMA, trajectory.py).
DP1: every condition reads with the same budget_tokens from configs/base.yaml.
"""

from __future__ import annotations

import time

from agentmem.eval.failures import tag_failure
from agentmem.eval.judge import Judge
from agentmem.eval.recall import evidence_metrics
from agentmem.harness.prompts import PROMPT_VERSION, build_messages
from agentmem.llm.client import LLMClient
from agentmem.logging.trajectory import TrajectoryWriter
from agentmem.memory import make_memory
from agentmem.tasks.custom_ds import ModeBEpisode, grade_answer
from agentmem.tasks.longmemeval import ModeAItem
from agentmem.utils.seeds import derive_seed, set_seed


def _base_record(mode: str, task: str, condition: str, item_id: str, seed: int) -> dict:
    return {
        "run_id": f"{task}-{condition}-{item_id}-s{seed}",
        "mode": mode, "task": task, "condition": condition,
        "item_id": item_id, "seed": seed, "prompt_version": PROMPT_VERSION,
    }


def _meter(llm: LLMClient) -> tuple[int, int, int]:
    """Cumulative (input, output, calls) on the shared client, including the
    write-path/overhead calls that _ask() cannot see. Deltas of this are the
    ONLY figures that satisfy Ch3's definition of token cost as 'total prompt
    and completion tokens per item'."""
    return (llm.cost.input_tokens, llm.cost.output_tokens, llm.cost.calls)


def _billed(before: tuple[int, int, int], after: tuple[int, int, int]) -> dict:
    return {
        "total_prompt_tokens": after[0] - before[0],
        "total_completion_tokens": after[1] - before[1],
        "total_llm_calls": after[2] - before[2],
    }


def _ask(memory, llm: LLMClient, question: str, budget: int) -> tuple[str, dict, dict]:
    """One budget-limited read + one generation. Returns (answer, ctx_info, cost)."""
    t0 = time.perf_counter()
    ctx = memory.read(question, budget_tokens=budget)
    assert ctx.tokens <= budget, f"DP1 violated: read {ctx.tokens} > budget {budget}"
    resp = llm.generate(build_messages(ctx.text, question))
    latency_ms = (time.perf_counter() - t0) * 1000
    cost = {
        "prompt_tokens": resp.input_tokens,
        "completion_tokens": resp.output_tokens,
        "memory_read_tokens": ctx.tokens,
        "latency_ms": round(latency_ms, 1),
    }
    return resp.text.strip(), {"provenance": ctx.provenance}, cost


def run_mode_a(
    items: list[ModeAItem],
    condition: str,
    llm: LLMClient,
    config: dict,
    writer: TrajectoryWriter,
    seed: int = 1,
    judge: Judge | None = None,
    condition_params: dict | None = None,
) -> int:
    """Returns number of new records written (skips already-done, resumable)."""
    budget = int(config["budgets"]["memory_read_tokens"])
    judge = judge or Judge()
    written = 0
    for item in items:
        if writer.is_done(condition, item.item_id, seed, "A"):
            continue
        set_seed(derive_seed(seed, condition, item.item_id))
        memory = make_memory(condition, condition_params, llm=llm)
        memory.reset()
        meter0 = _meter(llm)          # opens BEFORE ingest: catches write-path calls
        for turn in item.turns:
            memory.write(turn)
        answer, ctx_info, cost = _ask(memory, llm, item.question, budget)
        success = judge.grade(item.question, answer, item.answer)
        rec_m = evidence_metrics(ctx_info["provenance"], item.evidence_steps)
        stats = memory.stats()
        rec = {
            **_base_record("A", "longmemeval", condition, item.item_id, seed),
            "question_type": item.question_type,
            "question": item.question, "gold": item.answer, "prediction": answer,
            "success": bool(success),
            "failure_tag": tag_failure(success, answer, rec_m["evidence_recall"]),
            **rec_m,
            **cost,
            **_billed(meter0, _meter(llm)),
            "memory_written_tokens": stats.tokens_written,
            "overhead_calls": stats.overhead_calls,
            "memory_items": stats.items,
            "gbp_spent": round(llm.cost.gbp_spent, 6),
            "provenance": ctx_info["provenance"],  # FULL (was [:20] — truncation broke ordering audits)
        }
        writer.write(rec)
        written += 1
    return written


def _stale_hit(episode: ModeBEpisode, query, prediction: str) -> bool:
    """Did the model answer with an OLDER planted value for this fact?"""
    older_values = [
        t.metadata["value"] for t in episode.turns
        if t.metadata.get("kind") == "fact"
        and t.metadata.get("project") == query.project
        and t.metadata.get("attr") == query.attr
        and t.step < query.step
        and t.metadata["value"] != query.answer
    ]
    return any(grade_answer(prediction, v) for v in older_values)


def run_mode_b(
    episodes: list[ModeBEpisode],
    condition: str,
    llm: LLMClient,
    config: dict,
    writer: TrajectoryWriter,
    seed: int = 1,
    condition_params: dict | None = None,
) -> int:
    budget = int(config["budgets"]["memory_read_tokens"])
    written = 0
    for ep in episodes:
        memory = make_memory(condition, condition_params, llm=llm)
        memory.reset()
        set_seed(derive_seed(seed, condition, ep.episode_id))
        queries_by_step = {q.step: q for q in ep.queries}
        pending = [q for q in ep.queries
                   if not writer.is_done(condition, f"{ep.episode_id}:q{q.step}", seed, "B")]
        if not pending:
            continue
        # Overhead accrued since the previous query is amortised onto this query,
        # so the per-episode sum of total_prompt_tokens is the true episode spend.
        meter0 = _meter(llm)
        for turn in ep.turns:
            q = queries_by_step.get(turn.step)
            if q is not None:
                answer, ctx_info, cost = _ask(memory, llm, q.question, budget)
                success = grade_answer(answer, q.answer)
                rec_m = evidence_metrics(ctx_info["provenance"], q.evidence_steps)
                stale = (not success) and _stale_hit(ep, q, answer)
                stats = memory.stats()
                rec = {
                    **_base_record("B", "custom_ds", condition,
                                   f"{ep.episode_id}:q{q.step}", seed),
                    "question": q.question, "gold": q.answer, "prediction": answer,
                    "success": bool(success),
                    "failure_tag": tag_failure(success, answer,
                                               rec_m["evidence_recall"], stale_hit=stale),
                    "attr": q.attr, "evidence_distance": q.distance,
                    **rec_m,
                    **cost,
                    **_billed(meter0, _meter(llm)),
                    "memory_written_tokens": stats.tokens_written,
                    "overhead_calls": stats.overhead_calls,
                    "memory_items": stats.items,
                    "gbp_spent": round(llm.cost.gbp_spent, 6),
                    "provenance": ctx_info["provenance"],  # FULL (was [:20] — truncation broke ordering audits)
                }
                writer.write(rec)
                written += 1
                meter0 = _meter(llm)   # reset the window for the next query
            memory.write(turn)  # observation always enters memory (after any query)
    return written
