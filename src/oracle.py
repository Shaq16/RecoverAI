"""
The oracle (B*).

WHY THIS EXISTS
---------------
The obvious way to build this benchmark is to hand-write `optimal_action` for
each record. That is circular: the same intuitions would then be written into
the B2 rule table, so B2 would score near-perfectly by construction and the
whole comparison would be meaningless.

Instead the oracle *derives* the optimal action by searching the action space
against the frozen simulator with full knowledge of the hidden state. It is a
true upper bound. No policy can beat it, and the interesting number becomes
"what fraction of achievable value did this policy capture?".

The oracle obeys the same compliance gate as every policy, so a
compliance-trap record has a best *legal* action, not an illegal one.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from typing import Optional, Tuple

import yaml

from . import compliance
from .schema import (
    Observation, HiddenState,
    RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE, ABANDON,
    DELAY_GRID_HOURS, MAX_ATTEMPTS, MAX_HORIZON_HOURS,
)
from .simulator import success_probability

_ECON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "economics.yaml",
)


def load_economics(path: str = _ECON_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def action_cost(action: str, econ: dict) -> float:
    c = econ["costs"]
    return {
        RETRY: c["retry_attempt"],
        RETRY_LATER: c["retry_attempt"],
        REQUEST_PAYMENT_UPDATE: c["request_payment_update"],
        ABANDON: c["abandon"],
    }[action]


def _candidates():
    """(action, delay) pairs the oracle may consider."""
    out = [(ABANDON, 0)]
    for d in DELAY_GRID_HOURS:
        out.append((RETRY if d == 0 else RETRY_LATER, d))
    for d in DELAY_GRID_HOURS:
        out.append((REQUEST_PAYMENT_UPDATE, d))
    return out


def solve(obs: Observation, hidden: HiddenState,
          econ: Optional[dict] = None,
          cfg: Optional[dict] = None) -> Tuple[str, int, float]:
    """
    Return (optimal_action, optimal_delay_hours, expected_net_value).

    Value is expected contribution margin recovered minus intervention cost,
    in INR. ABANDON always has value 0, so any positive value means acting is
    worth it -- which is what makes the cost-trap slice bite.
    """
    econ = econ or load_economics()
    cfg = cfg or compliance.load_config()
    margin = econ["contribution_margin"]
    reward = obs.amount * margin

    cands = _candidates()

    @lru_cache(maxsize=None)
    def value(elapsed: int, attempts_used: int, updates_made: int) -> float:
        if attempts_used >= MAX_ATTEMPTS or elapsed >= MAX_HORIZON_HOURS:
            return 0.0
        state = replace(obs, elapsed_hours=elapsed, attempts_used=attempts_used,
                        update_requests_made=updates_made)
        best = 0.0  # ABANDON floor
        for action, delay in cands:
            if action == ABANDON:
                continue
            if compliance.check(state, action, delay, cfg).allowed is False:
                continue
            p = success_probability(state, hidden, action, delay)
            cost = action_cost(action, econ)
            landing = elapsed + delay + 1
            if action == REQUEST_PAYMENT_UPDATE:
                nxt = value(landing, attempts_used, updates_made + 1)
            else:
                nxt = value(landing, attempts_used + 1, updates_made)
            ev = p * reward + (1.0 - p) * nxt - cost
            if ev > best:
                best = ev
        return best

    # Root: pick the argmax explicitly so we can report the action.
    best_action, best_delay, best_ev = ABANDON, 0, 0.0
    for action, delay in cands:
        if action == ABANDON:
            continue
        if not compliance.check(obs, action, delay, cfg).allowed:
            continue
        p = success_probability(obs, hidden, action, delay)
        cost = action_cost(action, econ)
        landing = obs.elapsed_hours + delay + 1
        if action == REQUEST_PAYMENT_UPDATE:
            nxt = value(landing, obs.attempts_used, obs.update_requests_made + 1)
        else:
            nxt = value(landing, obs.attempts_used + 1, obs.update_requests_made)
        ev = p * reward + (1.0 - p) * nxt - cost
        if ev > best_ev + 1e-9:
            best_action, best_delay, best_ev = action, delay, ev

    value.cache_clear()
    return best_action, best_delay, best_ev


def annotate(obs: Observation, hidden: HiddenState,
             econ: Optional[dict] = None, cfg: Optional[dict] = None) -> HiddenState:
    """Fill optimal_action / optimal_delay_hours / oracle_ev / truly_recoverable."""
    action, delay, ev = solve(obs, hidden, econ, cfg)
    hidden.optimal_action = action
    hidden.optimal_delay_hours = delay
    hidden.oracle_ev = round(ev, 4)
    hidden.truly_recoverable = bool(action != ABANDON and ev > 0.0)
    return hidden
