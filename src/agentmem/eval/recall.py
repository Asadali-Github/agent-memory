"""Memory-recall accuracy from read() provenance — no extra LLM calls needed.

evidence_recall    = |gold evidence steps retrieved| / |gold evidence steps|
evidence_precision = |retrieved steps that are gold| / |retrieved steps|

Provenance entries may reference a single step ("step": int) or several
("steps": [ints]) for summaries/facts — both count as retrieving those steps.
"""

from __future__ import annotations


def provenance_steps(provenance: list[dict]) -> set[int]:
    steps: set[int] = set()
    for p in provenance:
        if isinstance(p.get("step"), int) and p["step"] >= 0:
            steps.add(p["step"])
        for s in p.get("steps", []) or []:
            if isinstance(s, int) and s >= 0:
                steps.add(s)
    return steps


def evidence_metrics(provenance: list[dict], evidence_steps: list[int]) -> dict:
    retrieved = provenance_steps(provenance)
    gold = set(evidence_steps)
    if not gold:
        return {"evidence_recall": None, "evidence_precision": None,
                "n_retrieved": len(retrieved)}
    hit = retrieved & gold
    return {
        "evidence_recall": len(hit) / len(gold),
        "evidence_precision": (len(hit) / len(retrieved)) if retrieved else 0.0,
        "n_retrieved": len(retrieved),
    }
