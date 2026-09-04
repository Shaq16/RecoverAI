"""
LLM client interface for B3, plus the observable-only payload builder.

TWO CLIENTS, ONE INTERFACE
--------------------------
`MockLLMClient` is deterministic and needs no API key, so the whole B3 path --
routing, gating, audit -- is testable and demoable offline. `AnthropicLLMClient`
calls Claude for real. Swapping them changes one constructor argument.

The mock is NOT a stub that returns a fixed answer, and it is NOT a disguised
oracle. It is a plausible reason-aware responder over the same observable
payload the real model sees, so a mock run exercises every gate and every audit
path that a live run does. Its determinism comes from SHA-256 over the canonical
payload, not Python's `hash()`, which is salted per process and would make runs
irreproducible.

PAYLOAD DISCIPLINE
------------------
`observable_payload` is an explicit allow-list, not `asdict(obs)` minus a few
keys. A field reaches the model only if it is named here, so adding a field to
Observation cannot silently leak it into a prompt. tests/test_b3_leakage.py
asserts the allow-list against the Observation dataclass in both directions.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schema import (
    Observation,
    ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
    DELAY_GRID_HOURS, MAX_ATTEMPTS,
    DO_NOT_HONOUR, CARD_EXPIRED, MANDATE_EXPIRED, AMOUNT_EXCEEDS_CAP,
    GATEWAY_TIMEOUT, ISSUER_DOWNTIME,
)

# Every field the model is permitted to see. Observation carries no ground
# truth (the schema guard proves it), but an explicit list means a future
# schema change cannot quietly widen what gets sent.
OBSERVABLE_FIELDS = (
    "payment_id", "amount", "currency", "decline_code", "payment_method",
    "customer_tenure_days", "prior_successes", "prior_failures",
    "days_into_billing_cycle", "day_of_month", "hour_of_day",
    "mandate_status", "mandate_cap", "recent_refund", "active_dispute",
    "pre_debit_notice_sent", "attempts_already_made", "subscription_plan_value",
    "attempts_used", "elapsed_hours", "update_requests_made",
)

VALID_ACTIONS = (RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE, ABANDON)


def observable_payload(obs: Observation) -> Dict:
    """The only thing an LLM client may be handed. Allow-list, not deny-list."""
    return {f: getattr(obs, f) for f in OBSERVABLE_FIELDS}


def payload_fingerprint(payload: Dict) -> str:
    """Stable hash of a payload. Process-independent, unlike hash()."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class LLMDecision:
    action: str
    delay_hours: int
    confidence: float
    rationale: str
    model: str = "mock"
    malformed: bool = False       # set when the response failed validation

    def to_dict(self) -> dict:
        return {
            "action": self.action, "delay_hours": self.delay_hours,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale, "model": self.model,
            "malformed": self.malformed,
        }


def _coerce(raw: dict, model: str) -> LLMDecision:
    """
    Validate a model response into an LLMDecision.

    A malformed response is not an exception -- it is a decision the router
    must be able to reject and audit. Marking it `malformed` lets B3 fall back
    to B2 and lets the audit trail show that it did.
    """
    action = str(raw.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        return LLMDecision(ABANDON, 0, 0.0, f"invalid action {action!r}",
                           model, malformed=True)
    try:
        delay = int(raw.get("delay_hours", 0))
    except (TypeError, ValueError):
        return LLMDecision(ABANDON, 0, 0.0, "non-integer delay_hours",
                           model, malformed=True)
    if delay < 0:
        return LLMDecision(ABANDON, 0, 0.0, "negative delay_hours",
                           model, malformed=True)
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    rationale = str(raw.get("rationale", ""))[:400]
    return LLMDecision(action, delay, conf, rationale, model)


class LLMClient:
    """Interface. `decide` maps an observable payload to a structured decision."""

    name = "base"

    def __init__(self):
        self.calls = 0
        self._cache: Dict[str, LLMDecision] = {}

    def decide(self, payload: Dict) -> LLMDecision:
        key = payload_fingerprint(payload)
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        decision = self._decide(payload)
        self._cache[key] = decision
        return decision

    def _decide(self, payload: Dict) -> LLMDecision:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """
    Deterministic offline stand-in.

    Encodes the kind of reasoning a competent model produces from these fields.
    It is deliberately not tuned against the simulator, and on some records it
    is simply wrong -- which is the honest condition for measuring whether
    routing to a model pays for itself.
    """

    name = "mock"

    def _decide(self, payload: Dict) -> LLMDecision:
        code = payload["decline_code"]
        cap = payload["mandate_cap"]
        amount = payload["amount"]
        dom = payload["day_of_month"]
        tenure = payload["customer_tenure_days"]
        succ, fail = payload["prior_successes"], payload["prior_failures"]
        status = payload["mandate_status"]

        # Deterministic jitter so confidence is not a constant, without
        # introducing run-to-run randomness.
        h = int(payload_fingerprint(payload)[:8], 16) / 0xFFFFFFFF

        if cap is not None and amount > cap:
            d = (REQUEST_PAYMENT_UPDATE, 0, 0.88,
                 "Amount exceeds the mandate cap; no retry can clear it. "
                 "The mandate has to be re-authorised at a higher limit.")
        elif status == "revoked" or code == "MANDATE_REVOKED":
            d = (ABANDON, 0, 0.72,
                 "Mandate revoked. A re-authorisation ask is legitimate but "
                 "rarely converts, and the expected value is thin.")
        elif code in (CARD_EXPIRED, MANDATE_EXPIRED) or status == "expired":
            d = (REQUEST_PAYMENT_UPDATE, 0, 0.90,
                 "Instrument or mandate has expired; this is a repair case, "
                 "not a retry case.")
        elif code == GATEWAY_TIMEOUT:
            d = (RETRY_LATER, 6, 0.85,
                 "Transient gateway failure. A short-delay retry usually "
                 "clears without troubling the customer.")
        elif code == ISSUER_DOWNTIME:
            d = (RETRY_LATER, 24, 0.80,
                 "Issuer-side outage. Wait out the window rather than burning "
                 "attempts against a bank that is down.")
        elif code == DO_NOT_HONOUR:
            spotless = tenure > 365 and (succ + fail) >= 10 and fail <= 1
            if spotless:
                d = (REQUEST_PAYMENT_UPDATE, 0, 0.66,
                     "A customer with a long spotless history does not usually "
                     "fail for want of funds. A silent mandate revocation fits "
                     "the evidence better, so ask them to re-authorise.")
            elif dom >= 25 or dom <= 3:
                d = (RETRY_LATER, 48, 0.62,
                     "Ambiguous decline near the salary window; funds are "
                     "likely to land shortly. Retry before escalating.")
            else:
                d = (REQUEST_PAYMENT_UPDATE, 0, 0.55,
                     "Ambiguous decline mid-cycle, when payroll is far away. "
                     "A repair ask dominates a blind retry here.")
        elif code == "INSUFFICIENT_FUNDS":
            if dom >= 25:
                d = (RETRY_LATER, max(24, min(168, (32 - dom) * 24)), 0.78,
                     "Insufficient funds close to month end; time the retry to "
                     "the expected salary credit.")
            elif dom <= 3:
                d = (RETRY_LATER, 24, 0.74,
                     "Insufficient funds just after month start; funds should "
                     "appear within a day.")
            else:
                d = (RETRY_LATER, 120, 0.58,
                     "Insufficient funds mid-cycle. Payroll is far off, so a "
                     "long delay beats an immediate retry.")
        elif code == "RISK_DECLINE":
            d = (REQUEST_PAYMENT_UPDATE, 0, 0.64,
                 "Risk decline. Repeating the same instrument will keep "
                 "failing; a different payment method may pass.")
        elif code == AMOUNT_EXCEEDS_CAP:
            d = (REQUEST_PAYMENT_UPDATE, 0, 0.86,
                 "Debit exceeds the authorised cap; the mandate needs raising.")
        else:
            d = (RETRY_LATER, 48, 0.50, "No strong signal; a single spaced "
                 "retry is the cheapest way to learn more.")

        action, delay, conf, rationale = d
        return _coerce(
            {"action": action, "delay_hours": delay,
             "confidence": round(min(0.97, conf + 0.06 * (h - 0.5)), 4),
             "rationale": rationale},
            self.name,
        )


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a payment-recovery analyst for an Indian subscription business. A \
recurring charge has failed. Decide the single best next action.

Actions:
  RETRY                  - re-attempt the debit immediately
  RETRY_LATER            - re-attempt after delay_hours
  REQUEST_PAYMENT_UPDATE - ask the customer to fix their payment method or mandate
  ABANDON                - stop pursuing this payment

What you should know about this domain:
- The observed decline_code is lossy. DO_NOT_HONOUR is a catch-all that hides \
insufficient funds, risk declines, issuer outages and silently revoked mandates.
- A retry cannot fix a structural failure. Expired cards, expired or revoked \
mandates, and amounts above the mandate cap need a customer repair, not another \
debit attempt.
- Indian salary credits cluster at the start of the month. An insufficient-funds \
decline on the 27th is worth waiting out; the same decline on the 12th is not.
- A long-tenured customer with a spotless record who suddenly hard-declines is \
more likely to have cancelled than to have run out of money. Do not read a good \
history as evidence that the money is recoverable.
- Recovering money is not always worth it. Low-value plans needing an expensive \
repair from a disengaged customer are better abandoned.

You recommend only. A separate deterministic engine enforces compliance, \
economics and retry budgets, and will override you. Do not assume your action \
will execute.

Set `confidence` to your genuine belief that this action is optimal. Calibration \
matters more than decisiveness: say 0.5 when the evidence is genuinely split."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string",
                   "enum": list(VALID_ACTIONS)},
        "delay_hours": {"type": "integer", "minimum": 0, "maximum": 336,
                        "description": "Hours to wait. 0 for RETRY/ABANDON."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string",
                      "description": "One or two sentences citing the fields "
                                     "that drove the decision."},
    },
    "required": ["action", "delay_hours", "confidence", "rationale"],
    "additionalProperties": False,
}


class AnthropicLLMClient(LLMClient):
    """
    Calls Claude for a single recovery decision.

    The system prompt is static and cached; only the per-record payload varies,
    so the cacheable prefix stays stable across the whole evaluation run.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 2048,
                 effort: str = "low", api_key: Optional[str] = None):
        super().__init__()
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The anthropic SDK is not installed. `pip install anthropic`, "
                "or use MockLLMClient to run B3 offline."
            ) from e
        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=api_key) if api_key \
            else anthropic.Anthropic()
        self.errors = 0

    def _decide(self, payload: Dict) -> LLMDecision:
        anthropic = self._anthropic
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": DECISION_SCHEMA},
                },
                messages=[{
                    "role": "user",
                    "content": "Failed recurring payment:\n"
                               + json.dumps(payload, indent=2, sort_keys=True),
                }],
            )
        except anthropic.NotFoundError as e:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, f"model/endpoint error: {e}",
                               self.model, malformed=True)
        except anthropic.RateLimitError as e:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, f"rate limited: {e}",
                               self.model, malformed=True)
        except anthropic.APIStatusError as e:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, f"api error {e.status_code}",
                               self.model, malformed=True)
        except anthropic.APIConnectionError as e:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, f"connection error: {e}",
                               self.model, malformed=True)

        if response.stop_reason == "refusal":
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, "model refused",
                               self.model, malformed=True)

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, "unparseable response",
                               self.model, malformed=True)
        return _coerce(raw, self.model)
