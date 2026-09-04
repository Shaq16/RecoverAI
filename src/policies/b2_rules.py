"""
B2: reason-aware deterministic rules. No LLM.

DESIGN
------
B2 is built as *the oracle's search run over a merchant's beliefs*. It uses the
same expectimax structure as src/oracle.py, but where the oracle reads the
hidden state, B2 substitutes an estimated success probability derived only from
the Observation. The gap between B2 and B* is therefore purely belief error,
which is the quantity the whole benchmark is trying to isolate.

This is deliberately a strong baseline. If B3 cannot beat a well-built rules
engine, that is the finding, and the README says so up front.

BELIEFS ARE NOT GROUND TRUTH
----------------------------
The numbers in `_P_UPDATE`, `_CODE_FIDELITY` and `_dnh_mixture` are beliefs
about the domain -- the sort of thing a merchant would estimate from their own
historical data. They are not read from the generator or the simulator, and
they are wrong in places. That is the point: a policy is allowed domain
knowledge, not per-record truth. The leakage guard makes the latter impossible
by construction.

The one genuinely clever move here is in `_dnh_mixture`: a long-tenured
customer with a spotless payment history who suddenly hard-declines as
DO_NOT_HONOUR is *more* likely to have cancelled their mandate than to have run
out of money. Weighting history positively is the trap that
`looks_alive_is_dead` is built to punish.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from .. import compliance, oracle
from ..schema import (
    Observation,
    ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
    DELAY_GRID_HOURS, MAX_ATTEMPTS, MAX_HORIZON_HOURS,
    INSUFFICIENT_FUNDS, ISSUER_DOWNTIME, GATEWAY_TIMEOUT, RISK_DECLINE,
    MANDATE_REVOKED, MANDATE_EXPIRED, CARD_EXPIRED, AMOUNT_EXCEEDS_CAP,
    DO_NOT_HONOUR,
)

# Confidence that an explicit (non-catch-all) decline code means what it says.
_CODE_FIDELITY = 0.88

# Believed success rate of a payment-method/mandate repair, given the true
# cause and a maximally responsive customer.
_P_UPDATE = {
    CARD_EXPIRED: 0.72,
    MANDATE_EXPIRED: 0.66,
    AMOUNT_EXCEEDS_CAP: 0.58,
    RISK_DECLINE: 0.30,
    MANDATE_REVOKED: 0.12,
    INSUFFICIENT_FUNDS: 0.18,
    ISSUER_DOWNTIME: 0.10,
    GATEWAY_TIMEOUT: 0.08,
}


# --------------------------------------------------------------------------
# Belief: what actually went wrong?
# --------------------------------------------------------------------------

def _engagement(obs: Observation) -> float:
    """Observable proxy for how likely this customer responds to an ask."""
    tenure = min(obs.customer_tenure_days / 540.0, 1.0)
    total = max(obs.prior_successes + obs.prior_failures, 1)
    history = obs.prior_successes / total
    e = 0.45 * tenure + 0.55 * history
    if obs.active_dispute:
        e *= 0.4
    if obs.recent_refund:
        e *= 0.8
    return max(0.0, min(1.0, e))


def _dnh_mixture(obs: Observation) -> Dict[str, float]:
    """
    Belief over true causes behind a DO_NOT_HONOUR.

    Structural causes are read straight off the observables where possible --
    a mandate cap below the amount is not ambiguous at all. What remains is a
    genuine three-way split between funds, risk and issuer downtime, plus the
    silent-revocation tail.
    """
    if obs.mandate_cap is not None and obs.amount > obs.mandate_cap:
        return {AMOUNT_EXCEEDS_CAP: 0.90, MANDATE_REVOKED: 0.05, RISK_DECLINE: 0.05}
    if obs.mandate_status == "revoked":
        return {MANDATE_REVOKED: 0.95, RISK_DECLINE: 0.05}
    if obs.mandate_status == "expired":
        return {MANDATE_EXPIRED: 0.90, MANDATE_REVOKED: 0.05, RISK_DECLINE: 0.05}

    mix = {INSUFFICIENT_FUNDS: 0.40, RISK_DECLINE: 0.28,
           ISSUER_DOWNTIME: 0.14, MANDATE_REVOKED: 0.18}

    # The looks_alive_is_dead correction. A customer who has paid 15 times
    # without a hiccup does not usually fail for want of INR 300; a silent
    # mandate revocation explains it better. Shift weight accordingly.
    total = obs.prior_successes + obs.prior_failures
    if (obs.customer_tenure_days > 365 and total >= 10
            and obs.prior_failures <= 1):
        mix = {INSUFFICIENT_FUNDS: 0.18, RISK_DECLINE: 0.14,
               ISSUER_DOWNTIME: 0.10, MANDATE_REVOKED: 0.58}

    return mix


def belief(obs: Observation) -> Dict[str, float]:
    """P(true_reason | observation). Sums to 1."""
    code = obs.decline_code
    if code == DO_NOT_HONOUR:
        mix = _dnh_mixture(obs)
    else:
        mix = {code: _CODE_FIDELITY}
        residual = _dnh_mixture(obs)
        scale = (1.0 - _CODE_FIDELITY) / max(sum(residual.values()), 1e-9)
        for r, p in residual.items():
            mix[r] = mix.get(r, 0.0) + p * scale

    # Hard observable overrides: these are facts, not inferences.
    if obs.mandate_status == "revoked":
        mix = {MANDATE_REVOKED: 1.0}
    elif obs.mandate_status == "expired" and code != CARD_EXPIRED:
        mix = {MANDATE_EXPIRED: 1.0}

    z = sum(mix.values())
    return {r: p / z for r, p in mix.items()}


# --------------------------------------------------------------------------
# Belief: when will the money be there?
# --------------------------------------------------------------------------

def _p_funded_by(obs: Observation, landing: int) -> float:
    """
    Believed P(funds available by `landing` hours), from the salary cycle.

    day_of_month is observable, and Indian salary credits cluster at month
    start. A mid-month insufficient-funds decline is a much worse retry
    candidate than one on the 27th, and this is where reason-aware timing
    earns most of its money over B1's fixed 24h interval.
    """
    dom = obs.day_of_month
    if dom >= 25:
        expected = (32 - dom) * 24
    elif dom <= 3:
        expected = 18
    else:
        expected = (31 - dom) * 24

    # Some accounts get funded early from a transfer unrelated to payroll.
    windfall = 0.22 * min(1.0, landing / 48.0)

    if landing >= expected:
        payroll = 0.85
    elif landing >= expected * 0.6:
        payroll = 0.30 * (landing / max(expected, 1))
    else:
        payroll = 0.0

    return max(0.0, min(0.95, windfall + (1.0 - windfall) * payroll))


def _staleness(elapsed: int) -> float:
    if elapsed <= 72:
        return 1.0
    return max(0.55, 1.0 - 0.0018 * (elapsed - 72))


def _p_retry(obs: Observation, bel: Dict[str, float], delay: int) -> float:
    landing = obs.elapsed_hours + delay
    p = 0.0
    for reason, w in bel.items():
        if reason == GATEWAY_TIMEOUT:
            p += w * (0.88 if landing >= 1 else 0.60)
        elif reason == ISSUER_DOWNTIME:
            # Outages are believed to be short; almost all clear within a day.
            p += w * (0.90 * min(1.0, landing / 20.0) if landing > 0 else 0.06)
        elif reason == INSUFFICIENT_FUNDS:
            p += w * 0.86 * _p_funded_by(obs, landing)
        elif reason == RISK_DECLINE:
            p += w * 0.06
        # structural causes contribute nothing: a retry cannot fix them
    return p * _staleness(landing) * (0.97 ** obs.attempts_used)


def _p_update(obs: Observation, bel: Dict[str, float], delay: int) -> float:
    landing = obs.elapsed_hours + delay
    resp = _engagement(obs)
    base = sum(w * _P_UPDATE.get(r, 0.10) for r, w in bel.items())
    return max(0.0, min(0.95, base * (0.35 + 0.65 * resp) * _staleness(landing)))


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------

def _candidates() -> List[Tuple[str, int]]:
    out = []
    for d in DELAY_GRID_HOURS:
        out.append((RETRY if d == 0 else RETRY_LATER, d))
    for d in DELAY_GRID_HOURS:
        out.append((REQUEST_PAYMENT_UPDATE, d))
    return out


class B2Rules:
    """
    Expectimax over the policy's own beliefs, under the real compliance gate.

    Returns a ranked candidate list, so if the gate denies the preferred action
    the harness can fall through to the next-best legal one rather than losing
    the episode. Knowing the rules is part of being a good policy.
    """

    NAME = "B2 rules"

    def __init__(self, econ: dict = None, cfg: dict = None, top_k: int = 4):
        self.econ = econ or oracle.load_economics()
        self.cfg = cfg or compliance.load_config()
        self.margin = self.econ["contribution_margin"]
        self.top_k = top_k
        self._cands = _candidates()

    def _rank(self, obs: Observation) -> List[Tuple[float, str, int]]:
        bel = belief(obs)
        reward = obs.amount * self.margin
        memo: Dict[tuple, float] = {}

        def value(elapsed: int, attempts: int, updates: int, depth: int) -> float:
            if depth <= 0 or attempts >= MAX_ATTEMPTS or elapsed >= MAX_HORIZON_HOURS:
                return 0.0
            key = (elapsed, attempts, updates, depth)
            if key in memo:
                return memo[key]
            state = replace(obs, elapsed_hours=elapsed, attempts_used=attempts,
                            update_requests_made=updates)
            best = 0.0                     # the ABANDON floor: never act at a loss
            for action, delay in self._cands:
                if not compliance.check(state, action, delay, self.cfg).allowed:
                    continue
                if action == REQUEST_PAYMENT_UPDATE:
                    p = _p_update(state, bel, delay)
                    nxt = value(elapsed + delay + 1, attempts, updates + 1, depth - 1)
                else:
                    p = _p_retry(state, bel, delay)
                    nxt = value(elapsed + delay + 1, attempts + 1, updates, depth - 1)
                ev = p * reward + (1.0 - p) * nxt - oracle.action_cost(action, self.econ)
                if ev > best:
                    best = ev
            memo[key] = best
            return best

        scored = []
        for action, delay in self._cands:
            # Only legal actions are ranked. The oracle searches under the same
            # constraint, so B2 is not being handed an unfair menu -- and a
            # policy that knows the rulebook should not be proposing debits it
            # is not allowed to make.
            if not compliance.check(obs, action, delay, self.cfg).allowed:
                continue
            if action == REQUEST_PAYMENT_UPDATE:
                p = _p_update(obs, bel, delay)
                nxt = value(obs.elapsed_hours + delay + 1, obs.attempts_used,
                            obs.update_requests_made + 1, 3)
            else:
                p = _p_retry(obs, bel, delay)
                nxt = value(obs.elapsed_hours + delay + 1, obs.attempts_used + 1,
                            obs.update_requests_made, 3)
            ev = p * reward + (1.0 - p) * nxt - oracle.action_cost(action, self.econ)
            scored.append((ev, action, delay))

        scored.sort(key=lambda t: -t[0])
        return scored

    def propose(self, obs: Observation):
        scored = self._rank(obs)

        # Economic gate: acting must beat stopping. This is what makes the
        # cost_trap slice answerable -- a recoverable payment is not always a
        # payment worth recovering.
        viable = [(ev, a, d) for ev, a, d in scored if ev > 0.0]
        if not viable:
            return [(ABANDON, 0)]

        out = [(a, d) for _, a, d in viable[: self.top_k]]
        out.append((ABANDON, 0))
        return out
