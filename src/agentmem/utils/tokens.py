"""Token counting.

Model-agnostic approximation (~4 chars/token) so the harness runs anywhere.
If `tiktoken` is installed it is used instead (closer for most BPE vocabs).
The SAME counter is used for every condition, so DP1 budget comparisons are
fair even if the absolute count differs from the served model's tokenizer.
"""

from __future__ import annotations

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        # disallowed_special=() : count special-token strings (e.g. literal
        # "<|endoftext|>" inside LongMemEval haystacks) as plain text instead
        # of raising — discovered in the real CERES run, 20 Jul 2026.
        return len(_ENC.encode(text, disallowed_special=()))

except ImportError:  # pragma: no cover - environment dependent

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        # ~4 chars/token is a standard approximation for English BPE.
        return max(1, round(len(text) / 4))


def truncate_to_tokens(text: str, budget: int) -> str:
    """Truncate text so count_tokens(result) <= budget (approximate, safe)."""
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text
    # Binary search on character length.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
