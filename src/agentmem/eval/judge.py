"""Answer grading.

Mode B: programmatic (grade via tasks.custom_ds.grade_answer) — no judge.
Mode A (LongMemEval): normalised-containment match by default; optional
LLM-as-judge (fixed judge model, frozen rubric) for free-form answers. Judge
calls are logged with their own tracker and NEVER charged to the condition.
"""

from __future__ import annotations

import re

from agentmem.llm.client import LLMClient

JUDGE_RUBRIC = (
    "You are grading a model answer against a gold answer.\n"
    "QUESTION: {question}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {prediction}\n\n"
    "Is the model answer factually correct and consistent with the gold "
    "answer? Reply with exactly one word: CORRECT or INCORRECT."
)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> bool:
    """Normalised containment either way (LongMemEval golds are short spans)."""
    p, g = _norm(prediction), _norm(gold)
    if not g:
        return False
    return g in p or (bool(p) and p in g)


class Judge:
    def __init__(self, llm: LLMClient | None = None):
        """llm=None -> pure string matching (pilot / mock mode)."""
        self.llm = llm
        self.calls = 0

    def grade(self, question: str, prediction: str, gold: str) -> bool:
        if exact_match(prediction, gold):
            return True
        if self.llm is None:
            return False
        resp = self.llm.generate(
            [{"role": "user", "content": JUDGE_RUBRIC.format(
                question=question, gold=gold, prediction=prediction)}],
            max_tokens=5,
        )
        self.calls += 1
        return resp.text.strip().upper().startswith("CORRECT")
