"""
Deterministic compliance gate.

Contract: the LLM recommends, this module decides whether the recommendation
is permitted. It runs identically for B1, B2 and B3 and for the oracle, so
"the oracle cheated" is never a valid criticism of a compliance-trap result.

Every denial returns a machine-readable reason string for the audit trail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml

from .schema import (
    Observation, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE, ABANDON,
)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "compliance.yaml",
)


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _v(node):
    """Unwrap a {value:, verified:, source:} node."""
    return node["value"] if isinstance(node, dict) and "value" in node else node


@dataclass
class ComplianceResult:
    allowed: bool
    reason: Optional[str] = None   # None when allowed

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason}


ALLOWED = ComplianceResult(True, None)


def check(obs: Observation, action: str, delay_hours: int,
          cfg: Optional[dict] = None) -> ComplianceResult:
    """Return whether `action` is permitted from state `obs`."""
    cfg = cfg or load_config()

    if action == ABANDON:
        return ALLOWED  # always permitted to stop

    total_attempts = obs.attempts_already_made + obs.attempts_used
    landing_hour = (obs.hour_of_day + delay_hours) % 24
    landing_elapsed = obs.elapsed_hours + delay_hours

    # ---------------- horizon ----------------
    rl = cfg["retry_limits"]
    max_days = _v(rl["max_days_from_first_failure"])
    if landing_elapsed > max_days * 24:
        return ComplianceResult(False, "beyond_max_days_from_first_failure")

    # ---------------- messaging ----------------
    if action == REQUEST_PAYMENT_UPDATE:
        msg = cfg["customer_messaging"]
        window = msg["allowed_hours"]
        if not (window["start"] <= landing_hour < window["end"]):
            return ComplianceResult(False, "outside_commercial_messaging_window")
        if obs.update_requests_made >= _v(msg["max_messages_per_episode"]):
            return ComplianceResult(False, "messaging_cap_exceeded")
        return ALLOWED

    # ---------------- debit attempts ----------------
    if action in (RETRY, RETRY_LATER):
        if total_attempts >= _v(rl["max_attempts_per_cycle"]):
            return ComplianceResult(False, "retry_cap_exceeded")
        if delay_hours < _v(rl["min_hours_between_attempts"]) and obs.attempts_used > 0:
            return ComplianceResult(False, "min_interval_between_attempts")

        ms = cfg["mandate_state"]
        if _v(ms["block_debit_when_revoked"]) and obs.mandate_status == "revoked":
            return ComplianceResult(False, "mandate_revoked")
        if _v(ms["block_debit_when_expired"]) and obs.mandate_status == "expired":
            return ComplianceResult(False, "mandate_expired")
        if (_v(ms["block_debit_above_mandate_cap"]) and obs.mandate_cap is not None
                and obs.amount > obs.mandate_cap):
            return ComplianceResult(False, "amount_exceeds_mandate_cap")

        if _v(cfg["disputes"]["block_debit_when_active_dispute"]) and obs.active_dispute:
            return ComplianceResult(False, "active_dispute")

        pdn = cfg["pre_debit_notification"]
        if _v(pdn["required_before_debit"]) and not obs.pre_debit_notice_sent:
            return ComplianceResult(False, "pre_debit_notice_missing")
        if obs.amount > _v(pdn["afa_threshold_inr"]):
            return ComplianceResult(False, "afa_required_above_threshold")

        return ALLOWED

    return ComplianceResult(False, "unknown_action")


def unverified_constants(cfg: Optional[dict] = None) -> list:
    """List every constant still marked verified: false. Printed by run_all."""
    cfg = cfg or load_config()
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            if "verified" in node:
                if not node["verified"]:
                    out.append(".".join(path))
                return
            for k, v in node.items():
                walk(v, path + [k])

    walk(cfg, [])
    return out
