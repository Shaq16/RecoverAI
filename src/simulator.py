"""
The simulator. This is the single source of truth for "would this action have
worked?", and every policy and the oracle are scored against it.

FROZEN CONTRACT
---------------
`success_probability` was written and committed BEFORE any policy (B1/B2/B3)
existed, and must not be edited in response to policy results. Tuning the
world model after seeing scores is training on the test set.

Any change to this file after the freeze must be recorded in
results/SIMULATOR_CHANGELOG.md with a reason.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .schema import (
    Observation, HiddenState,
    RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE, ABANDON,
    INSUFFICIENT_FUNDS, ISSUER_DOWNTIME, GATEWAY_TIMEOUT, RISK_DECLINE,
    MANDATE_REVOKED, MANDATE_EXPIRED, CARD_EXPIRED, AMOUNT_EXCEEDS_CAP,
)

SIMULATOR_VERSION = "1.0.0-frozen-2026-09-01"


# --------------------------------------------------------------------------
# Success model
# --------------------------------------------------------------------------

def _staleness(elapsed_hours: int) -> float:
    """Customers disengage. A recovery on day 12 is worth less than on day 1."""
    if elapsed_hours <= 72:
        return 1.0
    return max(0.55, 1.0 - 0.0018 * (elapsed_hours - 72))


def _attempt_fatigue(attempts_used: int) -> float:
    """Repeated hard declines make issuers less willing, not more."""
    return 0.97 ** attempts_used


def _engagement(obs: Observation) -> float:
    """0-1 proxy for how likely this customer is to respond to an ask."""
    tenure = min(obs.customer_tenure_days / 540.0, 1.0)
    history = obs.prior_successes / max(obs.prior_successes + obs.prior_failures, 1)
    e = 0.45 * tenure + 0.55 * history
    if obs.active_dispute:
        e *= 0.4
    if obs.recent_refund:
        e *= 0.8
    return max(0.0, min(1.0, e))


def success_probability(obs: Observation, hidden: HiddenState,
                        action: str, delay_hours: int) -> float:
    """P(action succeeds | true world state). Deterministic, no RNG."""
    if action == ABANDON:
        return 0.0

    landing = obs.elapsed_hours + delay_hours
    r = hidden.true_reason

    # ---- payment-method / mandate repair path ----
    if action == REQUEST_PAYMENT_UPDATE:
        base = {
            CARD_EXPIRED: 0.72,
            MANDATE_EXPIRED: 0.66,
            AMOUNT_EXCEEDS_CAP: 0.58,
            MANDATE_REVOKED: 0.12,     # they chose to leave; rarely comes back
            RISK_DECLINE: 0.30,        # a different instrument may pass
            INSUFFICIENT_FUNDS: 0.18,  # a new card does not create money
            ISSUER_DOWNTIME: 0.10,     # pointless, the bank is just down
            GATEWAY_TIMEOUT: 0.08,     # pointless, it was transient
        }[r]
        p = base * (0.35 + 0.65 * hidden.update_responsiveness)
        return max(0.0, min(0.95, p * _staleness(landing)))

    # ---- debit retry path ----
    if r == GATEWAY_TIMEOUT:
        p = 0.88 if landing >= 1 else 0.60

    elif r == ISSUER_DOWNTIME:
        over = landing >= (hidden.outage_duration_hours or 0)
        p = 0.90 if over else 0.06

    elif r == INSUFFICIENT_FUNDS:
        funded = landing >= (hidden.funds_return_hours or 0)
        p = 0.86 if funded else 0.07

    elif r == RISK_DECLINE:
        p = 0.06  # retrying a risk decline is mostly theatre

    elif r in (MANDATE_REVOKED, MANDATE_EXPIRED, AMOUNT_EXCEEDS_CAP, CARD_EXPIRED):
        p = 0.0   # structurally impossible; compliance also blocks these

    else:
        p = 0.0

    p *= _staleness(landing) * _attempt_fatigue(obs.attempts_used)
    return max(0.0, min(0.97, p))


# --------------------------------------------------------------------------
# Episode execution
# --------------------------------------------------------------------------

@dataclass
class StepOutcome:
    action: str
    delay_hours: int
    allowed: bool
    denial_reason: Optional[str]
    succeeded: bool
    elapsed_hours_after: int
    p_used: float


@dataclass
class EpisodeResult:
    payment_id: str
    recovered: bool
    amount: float
    steps: List[StepOutcome] = field(default_factory=list)
    stopped_reason: str = ""

    @property
    def debit_attempts(self) -> int:
        return sum(1 for s in self.steps
                   if s.allowed and s.action in (RETRY, RETRY_LATER))

    @property
    def update_requests(self) -> int:
        return sum(1 for s in self.steps
                   if s.allowed and s.action == REQUEST_PAYMENT_UPDATE)


def resolve(obs: Observation, hidden: HiddenState, action: str,
            delay_hours: int, rng: random.Random) -> bool:
    """Sample the outcome of one action."""
    return rng.random() < success_probability(obs, hidden, action, delay_hours)
