"""
Schema for RecoverAI.

Two layers, kept strictly separate:
  - Observation : everything a policy is allowed to see.
  - HiddenState : ground truth used ONLY by the simulator and the oracle.

If a policy ever touches HiddenState, the benchmark is void.
tests/test_no_leakage.py enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

RETRY = "RETRY"                                    # retry immediately
RETRY_LATER = "RETRY_LATER"                        # retry after delay_hours
REQUEST_PAYMENT_UPDATE = "REQUEST_PAYMENT_UPDATE"  # ask customer to fix method/mandate
ABANDON = "ABANDON"                                # stop, write off

ACTIONS = [RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE, ABANDON]

# Delay grid used by the oracle search and available to policies.
DELAY_GRID_HOURS = [0, 6, 12, 24, 48, 72, 120, 168]

# Episode limits (also enforced by the compliance engine).
MAX_ATTEMPTS = 4
MAX_HORIZON_HOURS = 336  # 14 days


# --------------------------------------------------------------------------
# Failure reasons
# --------------------------------------------------------------------------

# TRUE reasons (hidden). What is actually wrong with the payment.
INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
ISSUER_DOWNTIME = "ISSUER_DOWNTIME"
GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
RISK_DECLINE = "RISK_DECLINE"
MANDATE_REVOKED = "MANDATE_REVOKED"
MANDATE_EXPIRED = "MANDATE_EXPIRED"
CARD_EXPIRED = "CARD_EXPIRED"
AMOUNT_EXCEEDS_CAP = "AMOUNT_EXCEEDS_CAP"

TRUE_REASONS = [
    INSUFFICIENT_FUNDS, ISSUER_DOWNTIME, GATEWAY_TIMEOUT, RISK_DECLINE,
    MANDATE_REVOKED, MANDATE_EXPIRED, CARD_EXPIRED, AMOUNT_EXCEEDS_CAP,
]

# OBSERVED decline codes. Deliberately lossy: several true reasons collapse
# into DO_NOT_HONOUR, which is exactly what happens in production.
DO_NOT_HONOUR = "DO_NOT_HONOUR"

OBSERVED_CODES = TRUE_REASONS + [DO_NOT_HONOUR]


# --------------------------------------------------------------------------
# Adversarial slice tags
# --------------------------------------------------------------------------

SLICE_ORDINARY = "ordinary"
SLICE_LOOKS_ALIVE_IS_DEAD = "looks_alive_is_dead"
SLICE_COMPLIANCE_TRAP = "compliance_trap"
SLICE_COST_TRAP = "cost_trap"
SLICE_AMBIGUOUS_DNH = "ambiguous_dnh"

SLICES = [
    SLICE_ORDINARY, SLICE_LOOKS_ALIVE_IS_DEAD, SLICE_COMPLIANCE_TRAP,
    SLICE_COST_TRAP, SLICE_AMBIGUOUS_DNH,
]


@dataclass
class Observation:
    """Everything a policy may condition on. No hidden fields, ever."""
    payment_id: str
    amount: float                     # INR
    currency: str
    decline_code: str                 # observed, possibly DO_NOT_HONOUR
    payment_method: str               # upi_autopay | card | enach
    customer_tenure_days: int
    prior_successes: int
    prior_failures: int
    days_into_billing_cycle: int
    day_of_month: int
    hour_of_day: int                  # local, 0-23
    mandate_status: str               # active | expired | unknown
    mandate_cap: Optional[float]
    recent_refund: bool
    active_dispute: bool
    pre_debit_notice_sent: bool
    attempts_already_made: int        # before this episode begins
    subscription_plan_value: float    # monthly value, for LTV reasoning

    # Filled in during an episode; policies see their own history.
    attempts_used: int = 0
    elapsed_hours: int = 0
    update_requests_made: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HiddenState:
    """Ground truth. Simulator + oracle only."""
    true_reason: str
    # For INSUFFICIENT_FUNDS: hours until money is actually in the account.
    funds_return_hours: Optional[int] = None
    # For ISSUER_DOWNTIME: how long the outage lasts.
    outage_duration_hours: Optional[int] = None
    # Willingness to act on a payment-method-update request (0-1).
    update_responsiveness: float = 0.0
    # Filled by the oracle after search.
    optimal_action: Optional[str] = None
    optimal_delay_hours: Optional[int] = None
    oracle_ev: Optional[float] = None
    truly_recoverable: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Record:
    obs: Observation
    hidden: HiddenState
    slice_tag: str = SLICE_ORDINARY
    split: str = "train"

    def to_dict(self) -> dict:
        return {
            "obs": self.obs.to_dict(),
            "hidden": self.hidden.to_dict(),
            "slice_tag": self.slice_tag,
            "split": self.split,
        }

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(
            obs=Observation(**d["obs"]),
            hidden=HiddenState(**d["hidden"]),
            slice_tag=d["slice_tag"],
            split=d["split"],
        )
