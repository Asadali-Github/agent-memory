"""The single prompt template. IDENTICAL across all conditions — only the
{memory} slot content differs (that content is what each architecture chose
to spend the shared budget on). Changing this template invalidates
cross-condition comparability; version it if it must change.
"""

# CERES serves vLLM under Python 3.8, where PEP 585 generics (list[dict]) and
# PEP 604 unions (dict | None) raise TypeError at import time. This makes all
# annotations lazy strings so the module imports on 3.8. Do not remove.
from __future__ import annotations

PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using the MEMORY "
    "section below. If the memory contains multiple values for the same "
    "fact, the most recent one is correct. Answer concisely."
)

USER_TEMPLATE = """MEMORY:
{memory}

QUESTION:
{question}

Answer concisely:"""


def build_messages(memory_text: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            memory=memory_text or "(empty)", question=question)},
    ]
