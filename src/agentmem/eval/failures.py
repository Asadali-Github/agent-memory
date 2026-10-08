"""Rule-based failure tagging for failure-case clustering (RQ4).

Taxonomy (mutually exclusive, applied in order):
  success          — answered correctly
  no_answer        — empty/refusal output
  retrieval_miss   — NO gold evidence step made it into the read context
  partial_retrieval— some but not all gold evidence retrieved, answer wrong
  stale_fact       — (Mode B) answered with an OLDER planted value for the fact
  reasoning_failure— all evidence was in context, answer still wrong
"""

from __future__ import annotations


def tag_failure(
    success: bool,
    prediction: str,
    evidence_recall: float | None,
    stale_hit: bool = False,
) -> str:
    if success:
        return "success"
    if not prediction or not prediction.strip() or "i don't know" in prediction.lower():
        return "no_answer"
    if stale_hit:
        return "stale_fact"
    if evidence_recall is None:
        return "reasoning_failure"
    if evidence_recall == 0.0:
        return "retrieval_miss"
    if evidence_recall < 1.0:
        return "partial_retrieval"
    return "reasoning_failure"
