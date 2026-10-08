"""Token -> GBP cost accounting with a hard kill-switch (configs/base.yaml gbp_cap).

Prices are PER MILLION tokens in GBP. University HPC and mock are free; the
hosted-API rows are fallback paths. Update from provider price sheets before
any paid run and record the date in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (input_gbp_per_mtok, output_gbp_per_mtok) — verify before paid runs.
PRICE_SHEET: dict[str, tuple[float, float]] = {
    "mock": (0.0, 0.0),
    "university_hpc": (0.0, 0.0),
    "groq/llama-3.1-8b-instant": (0.04, 0.06),
    "groq/llama-3.3-70b-versatile": (0.47, 0.63),
    "openrouter/meta-llama/llama-3.1-8b-instruct": (0.03, 0.05),
}


class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend would exceed the configured GBP cap."""


@dataclass
class CostTracker:
    price_key: str
    gbp_cap: float = 30.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _prices: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        self._prices = PRICE_SHEET.get(self.price_key, (0.0, 0.0))

    @property
    def gbp_spent(self) -> float:
        pin, pout = self._prices
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.gbp_spent > self.gbp_cap:
            raise BudgetExceededError(
                f"Spend £{self.gbp_spent:.2f} exceeds cap £{self.gbp_cap:.2f} "
                f"after {self.calls} calls — halting run."
            )
