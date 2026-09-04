"""
B3: RecoverAI. Rules for the clear cases, an LLM for the ambiguous tail.

WHY ROUTE AT ALL
----------------
B2 already captures ~98% of oracle-achievable value on this benchmark, leaving
roughly INR 2.3k of regret across the test split. Invoking a model on every
record costs `costs.llm_decision` each time, which on 564 records is a
meaningful fraction of the entire remaining prize. Blanket-routing is therefore
provably the wrong architecture here, and the router is not an optimisation --
it is the whole thesis: AI earns its cost only where deterministic rules run
out of certainty.

WHEN B2 IS UNCERTAIN
--------------------
Ambiguity is measured between ACTION CLASSES, not between the top two
candidates. B2's ranked list is dominated by the same lever at adjacent delays
-- RETRY_LATER@24 versus RETRY_LATER@48 is a timing detail, not a dilemma. The
question worth paying a model to answer is "which lever do I pull": retry the
debit, ask the customer to repair, or stop. So the margin is taken between the
best debit option, the best repair option, and stopping.

Two pre-registered thresholds, fixed before B3 was first scored and not tuned
afterwards:
  AMBIGUITY_THRESHOLD -- route when the relative margin between the best and
      second-best action class is under 15%.
  MIN_STAKE_MULTIPLE  -- never pay for a decision worth less than 3x the cost
      of making it.

THE LLM DOES NOT EXECUTE
------------------------
Its recommendation passes three deterministic gates before it can be proposed:
compliance, economics, and retry budget. A denied recommendation falls back to
B2 rather than losing the episode, and every step of that is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import compliance, oracle
from ..schema import (
    Observation,
    ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
    MAX_ATTEMPTS, MAX_HORIZON_HOURS,
)
from .b2_rules import (
    B2Rules,
    belief as b2_belief,
    _p_retry as b2_p_retry,
    _p_update as b2_p_update,
)
from .llm_client import LLMClient, MockLLMClient, LLMDecision, observable_payload

# Pre-registered. See the module docstring.
AMBIGUITY_THRESHOLD = 0.15
MIN_STAKE_MULTIPLE = 3.0

DEBIT, REPAIR, STOP = "debit", "repair", "stop"


def _action_class(action: str) -> str:
    if action in (RETRY, RETRY_LATER):
        return DEBIT
    if action == REQUEST_PAYMENT_UPDATE:
        return REPAIR
    return STOP


@dataclass
class B3Decision:
    """One routed decision, complete enough to reconstruct why it happened."""
    payment_id: str
    elapsed_hours: int
    attempts_used: int
    b2_action: str
    b2_delay: int
    b2_ev: float
    margin: float                  # INR between best and second-best class
    relative_margin: float
    stake: float                   # INR in play on this decision
    ambiguous: bool
    llm_invoked: bool
    llm: Optional[LLMDecision] = None
    gate_results: Dict[str, str] = field(default_factory=dict)
    llm_accepted: bool = False
    final_action: str = ABANDON
    final_delay: int = 0

    def to_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "elapsed_hours": self.elapsed_hours,
            "attempts_used": self.attempts_used,
            "b2_proposal": {"action": self.b2_action, "delay_hours": self.b2_delay,
                            "ev_inr": round(self.b2_ev, 3)},
            "ambiguity": {"margin_inr": round(self.margin, 3),
                          "relative_margin": round(self.relative_margin, 4),
                          "stake_inr": round(self.stake, 3),
                          "ambiguous": self.ambiguous},
            "llm_invoked": self.llm_invoked,
            "llm_proposal": self.llm.to_dict() if self.llm else None,
            "gates": self.gate_results,
            "llm_accepted": self.llm_accepted,
            "final": {"action": self.final_action, "delay_hours": self.final_delay},
        }


class B3Router:
    """
    Rules-first router. Reuses B2 unchanged for both the confident path and the
    fallback path; the LLM is consulted only on the ambiguous, high-enough-stake
    tail, and never executes an action directly.
    """

    NAME = "B3 router"

    def __init__(self, econ: dict = None, cfg: dict = None,
                 client: Optional[LLMClient] = None,
                 b2: Optional[B2Rules] = None,
                 ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
                 min_stake_multiple: float = MIN_STAKE_MULTIPLE):
        self.econ = econ or oracle.load_economics()
        self.cfg = cfg or compliance.load_config()
        self.margin_rate = self.econ["contribution_margin"]
        self.llm_cost = self.econ["costs"]["llm_decision"]
        self.b2 = b2 or B2Rules(self.econ, self.cfg)
        self.client = client or MockLLMClient()
        self.ambiguity_threshold = ambiguity_threshold
        self.min_stake = min_stake_multiple * self.llm_cost
        # Keyed by (payment_id, elapsed_hours): unique along an episode path,
        # because elapsed_hours strictly increases at every step.
        self.decisions: Dict[Tuple[str, int], B3Decision] = {}

    # ---------------- ambiguity ----------------

    def _class_margins(self, scored: List[Tuple[float, str, int]]):
        """Best EV per action class, plus the margin between the top two."""
        best: Dict[str, Tuple[float, str, int]] = {}
        for ev, action, delay in scored:
            cls = _action_class(action)
            if cls not in best or ev > best[cls][0]:
                best[cls] = (ev, action, delay)
        # Stopping is always available and worth exactly zero.
        best.setdefault(STOP, (0.0, ABANDON, 0))
        if STOP in best and best[STOP][0] != 0.0:
            best[STOP] = (0.0, ABANDON, 0)

        ranked = sorted(best.values(), key=lambda t: -t[0])
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else (0.0, ABANDON, 0)
        margin = top[0] - second[0]
        relative = margin / max(abs(top[0]), 1.0)
        return top, margin, relative

    # ---------------- gates ----------------

    def _gate(self, obs: Observation, decision: LLMDecision,
              scored: List[Tuple[float, str, int]]) -> Tuple[bool, Dict[str, str]]:
        """
        Deterministic gates. The LLM recommends; these decide.

        Returns (accepted, {gate: verdict}). Every gate is recorded whether it
        passed or not, so the audit trail shows the full chain rather than only
        the first failure.
        """
        gates: Dict[str, str] = {}
        action, delay = decision.action, decision.delay_hours

        if decision.malformed:
            gates["schema"] = f"rejected: {decision.rationale}"
            return False, gates
        gates["schema"] = "ok"

        if action == ABANDON:
            gates["compliance"] = "ok (abandon always permitted)"
            gates["retry_budget"] = "ok"
            gates["economic"] = "ok (abandon has zero cost and zero value)"
            return True, gates

        # ---- retry budget ----
        total_attempts = obs.attempts_already_made + obs.attempts_used
        if action in (RETRY, RETRY_LATER) and total_attempts >= MAX_ATTEMPTS:
            gates["retry_budget"] = f"rejected: {total_attempts} attempts used"
            return False, gates
        if obs.elapsed_hours + delay > MAX_HORIZON_HOURS:
            gates["retry_budget"] = "rejected: beyond episode horizon"
            return False, gates
        gates["retry_budget"] = "ok"

        # ---- compliance ----
        verdict = compliance.check(obs, action, delay, self.cfg)
        if not verdict.allowed:
            gates["compliance"] = f"rejected: {verdict.reason}"
            return False, gates
        gates["compliance"] = "ok"

        # ---- economics ----
        # Scored under B2's belief model, which is the only estimate available
        # that does not consult ground truth. An action the merchant's own
        # economics say loses money does not execute on a model's say-so.
        ev, exact = self._economic_value(obs, action, delay, scored)
        basis = "" if exact else ", single-step lower bound (delay off B2 grid)"
        if ev <= 0.0:
            gates["economic"] = f"rejected: EV {ev:.2f} <= 0{basis}"
            return False, gates
        gates["economic"] = f"ok (EV {ev:.2f}{basis})"

        return True, gates

    def _economic_value(self, obs: Observation, action: str, delay: int,
                        scored: List[Tuple[float, str, int]]) -> Tuple[float, bool]:
        """
        EV of (action, delay) in INR under B2's belief model.

        B2 ranks only DELAY_GRID_HOURS, so a proposal at any other delay is
        simply absent from `scored`. Treating that absence as an economic
        rejection conflated "I cannot evaluate this" with "this loses money",
        and meant an off-grid proposal could never be assessed on its economics
        at all -- it was refused for a reason that was not economic.

        On a miss the proposal is evaluated directly, reusing B2's own belief
        functions so there remains exactly one economic model in the system.
        That estimate is single-step: it omits the continuation value B2's
        search adds for the branch where this action fails. Continuation value
        is non-negative -- B2's recursion floors at the ABANDON value of zero --
        so this is a LOWER BOUND on B2's full EV. Clearing the gate on the
        lower bound therefore implies clearing it under the full model, and the
        gate is strictly no weaker: it can only ever under-accept.

        Returns (ev, exact), where `exact` is True when the value came from
        B2's full search and False when it is the lower bound.
        """
        hit = next((e for e, a, d in scored if a == action and d == delay), None)
        if hit is not None:
            return hit, True

        bel = b2_belief(obs)
        if action == REQUEST_PAYMENT_UPDATE:
            p = b2_p_update(obs, bel, delay)
        else:
            p = b2_p_retry(obs, bel, delay)
        reward = obs.amount * self.margin_rate
        ev = p * reward - oracle.action_cost(action, self.econ)
        return ev, False

    # ---------------- policy interface ----------------

    def propose(self, obs: Observation):
        scored = self.b2._rank(obs)          # compliance-filtered, EV-sorted
        b2_proposal = self.b2.propose(obs)
        b2_action, b2_delay = b2_proposal[0]
        b2_ev = next((e for e, a, d in scored
                      if a == b2_action and d == b2_delay), 0.0)

        if scored:
            top, margin, relative = self._class_margins(scored)
            stake = max(top[0], 0.0)
        else:
            margin, relative, stake = 0.0, 1.0, 0.0

        ambiguous = (relative < self.ambiguity_threshold) and (stake >= self.min_stake)

        rec = B3Decision(
            payment_id=obs.payment_id, elapsed_hours=obs.elapsed_hours,
            attempts_used=obs.attempts_used,
            b2_action=b2_action, b2_delay=b2_delay, b2_ev=b2_ev,
            margin=margin, relative_margin=relative, stake=stake,
            ambiguous=ambiguous, llm_invoked=False,
            final_action=b2_action, final_delay=b2_delay,
        )

        if not ambiguous:
            self.decisions[(obs.payment_id, obs.elapsed_hours)] = rec
            return b2_proposal

        # ---- ambiguous: consult the model ----
        decision = self.client.decide(observable_payload(obs))
        rec.llm_invoked = True
        rec.llm = decision

        accepted, gates = self._gate(obs, decision, scored)
        rec.gate_results = gates
        rec.llm_accepted = accepted

        if not accepted:
            self.decisions[(obs.payment_id, obs.elapsed_hours)] = rec
            return b2_proposal

        rec.final_action = decision.action
        rec.final_delay = decision.delay_hours
        self.decisions[(obs.payment_id, obs.elapsed_hours)] = rec

        if decision.action == ABANDON:
            return [(ABANDON, 0)]
        # LLM choice first, then B2's ranking as fallback if the harness's own
        # gate denies it anyway.
        return [(decision.action, decision.delay_hours)] + b2_proposal
