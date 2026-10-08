"""Fixed-model LLM wrapper: one client, swappable backend, tokens/GBP/retry/kill-switch.

Backends
--------
- MockLLM              deterministic, zero-cost; develops/tests the whole pipeline on CPU.
- OpenAICompatibleLLM  any /chat/completions endpoint: vLLM on university HPC
                       (primary), Groq, OpenRouter (fallbacks). Base URL + key
                       come from env (.env), never from code.

The model is FIXED across all conditions (design invariant): the factory reads
it once from config; conditions cannot override it.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentmem.utils.costs import CostTracker
from agentmem.utils.tokens import count_tokens


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(ABC):
    """generate() is the ONLY way any component talks to the model."""

    def __init__(self, model: str, cost_tracker: CostTracker | None = None):
        self.model = model
        self.cost = cost_tracker or CostTracker(price_key="mock")

    @abstractmethod
    def _complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str: ...

    def generate(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512
    ) -> LLMResponse:
        prompt_text = "\n".join(m["content"] for m in messages)
        t0 = time.perf_counter()
        text = self._complete(messages, temperature, max_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp = LLMResponse(
            text=text,
            input_tokens=count_tokens(prompt_text),
            output_tokens=count_tokens(text),
            latency_ms=latency_ms,
        )
        self.cost.add(resp.input_tokens, resp.output_tokens)  # kill-switch lives here
        return resp


class MockLLM(LLMClient):
    """Deterministic stand-in with just enough behaviour to exercise the pipeline.

    - summarise prompts  -> keeps lines containing digits/proper nouns (facts survive)
    - extract prompts    -> emits '- fact' lines for fact-bearing sentences
    - answer prompts     -> returns the context line with max word overlap with the
                            question (so answer quality really depends on what the
                            memory architecture put in the context — conditions
                            differentiate even in mock mode)
    """

    def __init__(self, model: str = "mock"):
        super().__init__(model, CostTracker(price_key="mock", gbp_cap=float("inf")))

    def _complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        user = messages[-1]["content"]
        low = user.lower()
        if "summarize" in low or "summarise" in low:
            return self._keep_facts(user)
        if "extract" in low and "fact" in low:
            return self._extract_facts(user)
        return self._answer(user)

    @staticmethod
    def _turn_lines(text: str) -> list[str]:
        """Content lines look like '[role step N] ...' — strip the bracket prefix."""
        out = []
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("[") and "]" in ln:
                body = ln.split("]", 1)[1].strip()
                if body and re.search(r"\d|[A-Z][a-z]+ [A-Z][a-z]+|:", body):
                    out.append(body)
        return out

    def _keep_facts(self, text: str) -> str:
        return " ".join(self._turn_lines(text)[:40])[:2000] or "No salient facts."

    def _extract_facts(self, text: str) -> str:
        lines = self._turn_lines(text)
        return "\n".join(f"- {ln}" for ln in lines[:30]) or "- none"

    @staticmethod
    def _words(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    def _answer(self, prompt: str) -> str:
        # Question = last line ending in '?'; context = everything else.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        q_lines = [ln for ln in lines if ln.endswith("?")]
        question = q_lines[-1] if q_lines else (lines[-1] if lines else "")
        qw = self._words(question) - {"what", "which", "when", "where", "who", "how",
                                      "the", "a", "an", "is", "was", "did", "does", "of"}
        best, best_score = "I don't know.", 0
        for ln in lines:
            if ln == question:
                continue
            score = len(qw & self._words(ln))
            if score > best_score:
                best, best_score = ln, score
        return best


class OpenAICompatibleLLM(LLMClient):
    """Any OpenAI-compatible /chat/completions server (vLLM on HPC, Groq, OpenRouter)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        cost_tracker: CostTracker | None = None,
        max_retries: int = 4,
        timeout_s: int = 120,
    ):
        super().__init__(model, cost_tracker)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def _complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import requests  # local import: mock mode needs no network stack

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=self.timeout_s,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 — retry then surface
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


def make_llm(config: dict) -> LLMClient:
    """Factory. provider: mock | university_hpc | groq | openrouter."""
    provider = config.get("provider", "mock")
    model = config.get("model", "") or "mock"
    if provider == "mock":
        return MockLLM()

    cap = float(config.get("budgets", {}).get("gbp_cap", 30))
    tracker = CostTracker(price_key=f"{provider}/{model}"
                          if provider != "university_hpc" else "university_hpc",
                          gbp_cap=cap)
    if provider == "university_hpc":
        base_url = os.environ.get("HPC_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("HPC_API_KEY", "")
    elif provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.environ["GROQ_API_KEY"]
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ["OPENROUTER_API_KEY"]
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return OpenAICompatibleLLM(model, base_url, api_key, tracker)
