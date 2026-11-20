"""A hard USD cap on one run's paid requests, at the prices in harness/config.yaml and tools/*.yaml.

Every model turn, search and fetch takes the most it can cost from the cap before it is sent, and is refused
(BudgetStop) when that would pass the cap. Searches and fetches keep their price, failed or not. A model turn is
settled to its token count when the response reports usage. After one refusal every later request is refused too, so
the run stops. These are published prices, not invoices: a cached-token discount or a free allowance is not seen.
"""

from __future__ import annotations

import json
import threading
from typing import Any

# A token covers at least one byte of the text it encodes, so a request's input tokens cannot pass the UTF-8 length of
# its body; the margin covers what the service adds around messages and tool definitions.
FRAMING_MARGIN_TOKENS = 4_096


class BudgetStop(RuntimeError):
    """The spend cap is reached or a vendor account is out of credit; the run stops here."""


class Spend:
    def __init__(self, cap_usd: float, model_rates: dict[str, float]):
        self.cap_usd = cap_usd
        self.rates = model_rates
        self.spent_usd = 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def take(self, kind: str, usd: float) -> float:
        with self._lock:
            if self.stopped or self.spent_usd + usd > self.cap_usd:
                self.stopped = True
                raise BudgetStop(f"spend cap ${self.cap_usd:g} reached: ${self.spent_usd:.4f} spent, "
                                 f"this {kind} request may cost ${usd:.4f}")
            self.spent_usd += usd
        return usd

    def model_cost(self, input_tokens: int, output_tokens: int, *, worst: bool = False) -> float:
        long = worst or input_tokens > self.rates["long_context_tokens"]
        rate_in = self.rates["long_input_usd_per_million" if long else "input_usd_per_million"]
        rate_out = self.rates["long_output_usd_per_million" if long else "output_usd_per_million"]
        return (input_tokens * rate_in + output_tokens * rate_out) / 1e6

    def take_model(self, request: dict[str, Any]) -> float:
        tokens = len(json.dumps(request, default=str).encode()) + FRAMING_MARGIN_TOKENS
        return self.take("model", self.model_cost(tokens, request["max_output_tokens"], worst=True))

    def settle_model(self, held: float, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.spent_usd += self.model_cost(input_tokens, output_tokens) - held
