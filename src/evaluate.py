"""
Evaluation harness.

EXACT, NOT SAMPLED
------------------
`simulator.success_probability` is deterministic and the episode tree is small
(bounded by MAX_ATTEMPTS, MAX_HORIZON_HOURS and the messaging cap), so a
policy's expected value can be computed in closed form by recursing both the
success and failure branches. There is no RNG anywhere in this module.

This matters. An earlier Monte-Carlo version of the degeneracy check moved the
headline number by ~5 points depending only on the sampling seed, and reported
the wrong ranking between two baselines. Exact evaluation removes that entire
class of error: run it twice, get the same answer.

The only randomness in reported metrics is the bootstrap, which resamples
RECORDS to express dataset uncertainty -- the thing we actually want a
confidence interval on.

POLICY PROTOCOL
---------------
A policy is an object (or module) with:

    propose(obs: Observation) -> list[(action, delay_hours)]

returning candidates in preference order. The harness walks the list and takes
the first that clears the compliance gate, recording every denial in the audit
trail. This mirrors the architecture: the policy recommends, the gate decides.
A policy that proposes illegal actions is not crashed -- it simply loses the
value it would have earned, which is what makes the compliance_trap slice bite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import compliance, oracle
from .schema import (
    Observation, Record,
    ABANDON, MAX_ATTEMPTS, MAX_HORIZON_HOURS, SLICES,
    REQUEST_PAYMENT_UPDATE,
)
from .simulator import success_probability

MAX_STEPS = 12          # safety net; the schema limits bind first in practice
BOOTSTRAP_SEED = 20260903


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------

@dataclass
class AuditStep:
    """One decision, with everything needed to explain it after the fact."""
    step: int
    proposed: List[Tuple[str, int]]
    denials: List[Tuple[str, int, str]]      # (action, delay, reason)
    chosen: Optional[str]
    chosen_delay: Optional[int]
    p_success: float
    elapsed_hours: int
    attempts_used: int

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "proposed": [[a, d] for a, d in self.proposed],
            "denials": [[a, d, r] for a, d, r in self.denials],
            "chosen": self.chosen,
            "chosen_delay": self.chosen_delay,
            "p_success": round(self.p_success, 6),
            "elapsed_hours": self.elapsed_hours,
            "attempts_used": self.attempts_used,
        }


@dataclass
class RecordResult:
    payment_id: str
    slice_tag: str
    amount: float
    policy_ev: float
    oracle_ev: float
    root_action: str
    oracle_action: str
    denied_count: int
    audit: List[AuditStep] = field(default_factory=list)

    @property
    def root_action_matches_oracle(self) -> bool:
        return self.root_action == self.oracle_action

    def to_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "slice_tag": self.slice_tag,
            "amount": self.amount,
            "policy_ev": round(self.policy_ev, 4),
            "oracle_ev": round(self.oracle_ev, 4),
            "root_action": self.root_action,
            "oracle_action": self.oracle_action,
            "denied_count": self.denied_count,
            "audit": [s.to_dict() for s in self.audit],
        }


# --------------------------------------------------------------------------
# Exact single-record evaluation
# --------------------------------------------------------------------------

def _normalise(proposal) -> List[Tuple[str, int]]:
    """Accept a single (action, delay) or a ranked list of them."""
    if proposal is None:
        return [(ABANDON, 0)]
    if isinstance(proposal, tuple) and len(proposal) == 2 \
            and isinstance(proposal[0], str):
        return [proposal]
    return [(a, int(d)) for a, d in proposal]


def evaluate_record(rec: Record, policy, econ: dict, cfg: dict,
                    margin: float, collect_audit: bool = False) -> RecordResult:
    """
    Exact expected net value of `policy` on `rec`, in INR.

    Recurses both outcome branches rather than sampling. Value is expected
    contribution margin recovered minus intervention cost, so ABANDON scores
    exactly 0 and any positive number means acting was worth it.
    """
    hidden = rec.hidden
    reward = rec.obs.amount * margin
    audit: List[AuditStep] = []
    root: Dict[str, Optional[str]] = {"action": None}
    denied_total = [0]

    def walk(obs: Observation, step: int) -> float:
        if step >= MAX_STEPS \
                or obs.attempts_used >= MAX_ATTEMPTS \
                or obs.elapsed_hours >= MAX_HORIZON_HOURS:
            return 0.0

        proposed = _normalise(policy.propose(obs))
        denials: List[Tuple[str, int, str]] = []
        chosen: Optional[Tuple[str, int]] = None

        for action, delay in proposed:
            if action == ABANDON:
                chosen = (ABANDON, 0)
                break
            verdict = compliance.check(obs, action, delay, cfg)
            if verdict.allowed:
                chosen = (action, delay)
                break
            denials.append((action, delay, verdict.reason or "denied"))

        denied_total[0] += len(denials)
        if chosen is None:                      # every proposal was illegal
            chosen = (ABANDON, 0)

        action, delay = chosen
        if root["action"] is None:
            root["action"] = action

        if action == ABANDON:
            if collect_audit:
                audit.append(AuditStep(step, proposed, denials, ABANDON, 0, 0.0,
                                       obs.elapsed_hours, obs.attempts_used))
            return 0.0

        p = success_probability(obs, hidden, action, delay)
        if collect_audit:
            audit.append(AuditStep(step, proposed, denials, action, delay, p,
                                   obs.elapsed_hours, obs.attempts_used))

        landing = obs.elapsed_hours + delay + 1
        if action == REQUEST_PAYMENT_UPDATE:
            nxt = replace(obs, elapsed_hours=landing,
                          update_requests_made=obs.update_requests_made + 1)
        else:
            nxt = replace(obs, elapsed_hours=landing,
                          attempts_used=obs.attempts_used + 1)

        cost = oracle.action_cost(action, econ)
        return p * reward + (1.0 - p) * walk(nxt, step + 1) - cost

    ev = walk(replace(rec.obs), 0)

    return RecordResult(
        payment_id=rec.obs.payment_id,
        slice_tag=rec.slice_tag,
        amount=rec.obs.amount,
        policy_ev=ev,
        oracle_ev=rec.hidden.oracle_ev or 0.0,
        root_action=root["action"] or ABANDON,
        oracle_action=rec.hidden.optimal_action or ABANDON,
        denied_count=denied_total[0],
        audit=audit,
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class SliceMetrics:
    slice_tag: str
    n: int
    policy_value: float
    oracle_value: float
    root_agreement: float          # share of records matching the oracle's action
    abandon_rate: float
    denied_rate: float             # mean illegal proposals per record
    waste: float                   # INR burned on records the oracle abandons
    regret: float                  # oracle value not captured (INR)

    @property
    def share(self) -> float:
        if abs(self.oracle_value) < 1e-9:
            return 0.0
        return self.policy_value / self.oracle_value

    def to_dict(self) -> dict:
        return {
            "slice": self.slice_tag, "n": self.n,
            "policy_value": round(self.policy_value, 2),
            "oracle_value": round(self.oracle_value, 2),
            "share_of_oracle": round(self.share, 4),
            "root_agreement": round(self.root_agreement, 4),
            "abandon_rate": round(self.abandon_rate, 4),
            "denied_per_record": round(self.denied_rate, 4),
            "waste_inr": round(self.waste, 2),
            "regret_inr": round(self.regret, 2),
        }


@dataclass
class PolicyMetrics:
    name: str
    n: int
    policy_value: float
    oracle_value: float
    share: float
    ci_low: float
    ci_high: float
    root_agreement: float
    abandon_rate: float
    denied_rate: float
    waste: float
    regret: float
    per_slice: Dict[str, SliceMetrics]
    results: List[RecordResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy": self.name, "n": self.n,
            "policy_value": round(self.policy_value, 2),
            "oracle_value": round(self.oracle_value, 2),
            "share_of_oracle": round(self.share, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "root_agreement": round(self.root_agreement, 4),
            "abandon_rate": round(self.abandon_rate, 4),
            "denied_per_record": round(self.denied_rate, 4),
            "waste_inr": round(self.waste, 2),
            "regret_inr": round(self.regret, 2),
            "per_slice": {k: v.to_dict() for k, v in self.per_slice.items()},
        }


def _waste(rs: Sequence[RecordResult]) -> float:
    """
    Money spent chasing payments the oracle says to drop.

    This is the metric the cost_trap slice actually needs. That slice holds
    0.0% of total oracle value, so share-of-oracle is blind to it -- a policy
    can set fire to the entire slice and barely move the headline number.
    Waste is denominated in INR, so it cannot be hidden by a small denominator.
    """
    return -sum(r.policy_ev for r in rs
                if r.oracle_action == ABANDON and r.policy_ev < 0)


def _slice_metrics(tag: str, rs: Sequence[RecordResult]) -> SliceMetrics:
    n = len(rs)
    return SliceMetrics(
        slice_tag=tag, n=n,
        policy_value=sum(r.policy_ev for r in rs),
        oracle_value=sum(r.oracle_ev for r in rs),
        root_agreement=(sum(r.root_action_matches_oracle for r in rs) / n) if n else 0.0,
        abandon_rate=(sum(r.root_action == ABANDON for r in rs) / n) if n else 0.0,
        denied_rate=(sum(r.denied_count for r in rs) / n) if n else 0.0,
        waste=_waste(rs),
        regret=sum(r.oracle_ev - r.policy_ev for r in rs),
    )


def _bootstrap_ci(results: Sequence[RecordResult], iters: int = 2000,
                  seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """
    Percentile CI on share-of-oracle, resampling RECORDS with replacement.

    The uncertainty being expressed is "would another 1,200 draws from this
    generator give the same answer?" -- not sampling noise in the simulator,
    which exact evaluation has already eliminated.
    """
    n = len(results)
    if n == 0:
        return (0.0, 0.0)
    pol = [r.policy_ev for r in results]
    orc = [r.oracle_ev for r in results]
    rng = random.Random(seed)
    shares = []
    for _ in range(iters):
        p = o = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            p += pol[i]
            o += orc[i]
        shares.append(p / o if abs(o) > 1e-9 else 0.0)
    shares.sort()
    lo = shares[int(0.025 * len(shares))]
    hi = shares[min(int(0.975 * len(shares)), len(shares) - 1)]
    return (lo, hi)


def evaluate(recs: Sequence[Record], policy, name: Optional[str] = None,
             econ: Optional[dict] = None, cfg: Optional[dict] = None,
             bootstrap: int = 2000, collect_audit: bool = False) -> PolicyMetrics:
    """Score `policy` over `recs`. Deterministic apart from the record bootstrap."""
    econ = econ or oracle.load_economics()
    cfg = cfg or compliance.load_config()
    margin = econ["contribution_margin"]
    name = name or getattr(policy, "NAME", policy.__class__.__name__)

    results = [evaluate_record(r, policy, econ, cfg, margin, collect_audit)
               for r in recs]

    n = len(results)
    pol_v = sum(r.policy_ev for r in results)
    orc_v = sum(r.oracle_ev for r in results)
    lo, hi = _bootstrap_ci(results, bootstrap) if bootstrap else (0.0, 0.0)

    per_slice = {}
    for tag in SLICES:
        rs = [r for r in results if r.slice_tag == tag]
        if rs:
            per_slice[tag] = _slice_metrics(tag, rs)

    return PolicyMetrics(
        name=name, n=n, policy_value=pol_v, oracle_value=orc_v,
        share=(pol_v / orc_v if abs(orc_v) > 1e-9 else 0.0),
        ci_low=lo, ci_high=hi,
        root_agreement=sum(r.root_action_matches_oracle for r in results) / n if n else 0.0,
        abandon_rate=sum(r.root_action == ABANDON for r in results) / n if n else 0.0,
        denied_rate=sum(r.denied_count for r in results) / n if n else 0.0,
        waste=_waste(results),
        regret=orc_v - pol_v,
        per_slice=per_slice,
        results=results,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_table(metrics: Sequence[PolicyMetrics]) -> str:
    out = []
    w = 22
    out.append(f"{'policy':<{w}}{'% oracle':>10}{'95% CI':>16}{'agree':>8}"
               f"{'regret':>11}{'waste':>10}{'illegal':>9}")
    out.append("-" * (w + 64))
    for m in metrics:
        ci = f"[{100*m.ci_low:.1f}, {100*m.ci_high:.1f}]"
        out.append(
            f"{m.name:<{w}}{100*m.share:>9.1f}%{ci:>16}"
            f"{100*m.root_agreement:>7.0f}%{m.regret:>11,.0f}{m.waste:>10,.0f}"
            f"{m.denied_rate:>9.2f}"
        )
    return "\n".join(out)


def format_slice_table(metrics: Sequence[PolicyMetrics]) -> str:
    tags = [t for t in SLICES if any(t in m.per_slice for m in metrics)]
    out = []
    out.append(f"{'slice':<22}{'n':>5}{'oracle':>11}"
               + "".join(f"{m.name:>16}" for m in metrics))
    out.append("-" * (38 + 16 * len(metrics)))
    for t in tags:
        ref = next(m.per_slice[t] for m in metrics if t in m.per_slice)
        row = f"{t:<22}{ref.n:>5}{ref.oracle_value:>11,.0f}"
        for m in metrics:
            s = m.per_slice.get(t)
            row += f"{100*s.share:>15.1f}%" if s else f"{'-':>16}"
        out.append(row)
    return "\n".join(out)
