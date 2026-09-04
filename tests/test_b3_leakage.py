"""
B3-specific guards.

tests/test_no_leakage.py already AST-scans src/policies/, which covers
b3_router.py and llm_client.py. This file covers what an AST scan cannot: what
is actually inside the dict handed to the model at runtime, and whether the
deterministic gates really override the model rather than rubber-stamping it.
"""

from __future__ import annotations

import dataclasses

import pytest

from src import compliance, oracle
from src.evaluate import evaluate
from src.generate import build
from src.policies.b2_rules import B2Rules
from src.policies.b3_router import B3Router
from src.policies.llm_client import (
    LLMClient, LLMDecision, MockLLMClient, OBSERVABLE_FIELDS,
    observable_payload, payload_fingerprint, _coerce,
)
from src.schema import (
    Observation, HiddenState, Record,
    ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
)
from tests.test_no_leakage import HIDDEN_FIELDS


@pytest.fixture(scope="module")
def env():
    return oracle.load_economics(), compliance.load_config()


@pytest.fixture(scope="module")
def recs():
    return [r for r in build(300, 7) if r.split == "test"]


def _obs(**over):
    base = dict(
        payment_id="pay_00042", amount=499.0, currency="INR",
        decline_code="DO_NOT_HONOUR", payment_method="card",
        customer_tenure_days=800, prior_successes=20, prior_failures=0,
        days_into_billing_cycle=1, day_of_month=14, hour_of_day=11,
        mandate_status="active", mandate_cap=1500.0, recent_refund=False,
        active_dispute=False, pre_debit_notice_sent=True,
        attempts_already_made=0, subscription_plan_value=499.0,
    )
    base.update(over)
    return Observation(**base)


# ---------------------------------------------------------------------------
# The payload the model actually sees
# ---------------------------------------------------------------------------

def test_allowlist_matches_observation_exactly():
    """
    Both directions on purpose. A new Observation field must be added here
    deliberately, and a stale name here must not silently do nothing.
    """
    fields = {f.name for f in dataclasses.fields(Observation)}
    assert set(OBSERVABLE_FIELDS) == fields, (
        f"missing from allow-list: {fields - set(OBSERVABLE_FIELDS)}; "
        f"not on Observation: {set(OBSERVABLE_FIELDS) - fields}"
    )


def test_payload_carries_no_hidden_keys():
    assert not set(observable_payload(_obs())) & HIDDEN_FIELDS


def test_payload_carries_no_hidden_values():
    """
    Key-name checks miss a value smuggled under an innocent key, so compare
    against the actual ground truth of a real record.
    """
    rec = Record(
        obs=_obs(),
        hidden=HiddenState(true_reason="MANDATE_REVOKED", funds_return_hours=91,
                           outage_duration_hours=17, update_responsiveness=0.7331,
                           optimal_action=REQUEST_PAYMENT_UPDATE,
                           optimal_delay_hours=48, oracle_ev=123.456,
                           truly_recoverable=True),
    )
    payload_values = set(map(repr, observable_payload(rec.obs).values()))
    for name, value in dataclasses.asdict(rec.hidden).items():
        if value is None or isinstance(value, bool):
            continue          # booleans and None collide with observables benignly
        assert repr(value) not in payload_values, f"hidden {name}={value!r} in payload"


def test_router_hands_the_client_nothing_but_the_allowlist(env):
    """Spy on the real call path rather than trusting observable_payload alone."""
    econ, cfg = env
    seen = []

    class Spy(LLMClient):
        name = "spy"

        def _decide(self, payload):
            seen.append(payload)
            return LLMDecision(ABANDON, 0, 0.5, "spy")

    b3 = B3Router(econ, cfg, client=Spy(), ambiguity_threshold=1e9,
                  min_stake_multiple=0.0)  # force routing
    b3.propose(_obs())
    assert seen, "client was never called"
    for payload in seen:
        assert set(payload) == set(OBSERVABLE_FIELDS)
        assert not set(payload) & HIDDEN_FIELDS


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_mock_is_deterministic_across_instances():
    p = observable_payload(_obs())
    a, b = MockLLMClient()._decide(p), MockLLMClient()._decide(p)
    assert (a.action, a.delay_hours, a.confidence) == \
           (b.action, b.delay_hours, b.confidence)


def test_fingerprint_is_stable_and_order_independent():
    p = observable_payload(_obs())
    assert payload_fingerprint(p) == payload_fingerprint(dict(reversed(list(p.items()))))


def test_client_caches_repeat_payloads():
    c = MockLLMClient()
    p = observable_payload(_obs())
    c.decide(p); c.decide(p)
    assert c.calls == 1, "identical payloads should not be billed twice"


# ---------------------------------------------------------------------------
# The gates actually override the model
# ---------------------------------------------------------------------------

class _Fixed(LLMClient):
    name = "fixed"

    def __init__(self, decision):
        super().__init__()
        self._d = decision

    def _decide(self, payload):
        return self._d


def _forced(env, decision):
    """
    Force an invocation regardless of ambiguity or stake, so a gate test
    exercises the gate it names. Without min_stake_multiple=0 the stake gate
    can short-circuit first and the assertion tests nothing.
    """
    econ, cfg = env
    return B3Router(econ, cfg, client=_Fixed(decision),
                    ambiguity_threshold=1e9, min_stake_multiple=0.0)


def test_gate_rejects_compliance_violation(env):
    """Debit against a revoked mandate must never reach the proposal list."""
    b3 = _forced(env, LLMDecision(RETRY, 0, 0.99, "just retry it"))
    obs = _obs(mandate_status="revoked")
    proposal = b3.propose(obs)
    rec = b3.decisions[(obs.payment_id, obs.elapsed_hours)]
    assert rec.llm_invoked and not rec.llm_accepted
    assert "rejected" in rec.gate_results["compliance"]
    assert (RETRY, 0) not in proposal


def test_gate_rejects_retry_budget_exhaustion(env):
    b3 = _forced(env, LLMDecision(RETRY_LATER, 24, 0.99, "one more"))
    obs = _obs(attempts_already_made=4)
    b3.propose(obs)
    rec = b3.decisions[(obs.payment_id, obs.elapsed_hours)]
    assert not rec.llm_accepted
    assert "rejected" in rec.gate_results["retry_budget"]


def test_gate_rejects_malformed_output(env):
    b3 = _forced(env, _coerce({"action": "REFUND_EVERYTHING"}, "fixed"))
    obs = _obs()
    b3.propose(obs)
    rec = b3.decisions[(obs.payment_id, obs.elapsed_hours)]
    assert not rec.llm_accepted and "rejected" in rec.gate_results["schema"]


def test_gate_rejects_uneconomic_action(env):
    """A tiny plan needing an expensive repair: the economics must veto."""
    b3 = _forced(env, LLMDecision(REQUEST_PAYMENT_UPDATE, 0, 0.99, "ask them"))
    obs = _obs(amount=29.0, subscription_plan_value=29.0, decline_code="MANDATE_REVOKED",
               customer_tenure_days=25, prior_successes=0, prior_failures=3)
    b3.propose(obs)
    rec = b3.decisions[(obs.payment_id, obs.elapsed_hours)]
    assert rec.llm_invoked and not rec.llm_accepted
    assert "rejected" in rec.gate_results["economic"]


def test_rejected_llm_falls_back_to_b2_not_to_failure(env):
    econ, cfg = env
    b3 = _forced(env, LLMDecision(RETRY, 0, 0.99, "retry"))
    obs = _obs(mandate_status="revoked")
    assert b3.propose(obs) == B2Rules(econ, cfg).propose(obs)


# ---------------------------------------------------------------------------
# B3 reuses B2 rather than reimplementing it
# ---------------------------------------------------------------------------

def test_router_with_routing_disabled_is_exactly_b2(env, recs):
    econ, cfg = env
    never = B3Router(econ, cfg, client=MockLLMClient(), ambiguity_threshold=-1.0)
    b2 = evaluate(recs, B2Rules(econ, cfg), econ=econ, cfg=cfg, bootstrap=0)
    b3 = evaluate(recs, never, econ=econ, cfg=cfg, bootstrap=0)
    assert abs(b2.policy_value - b3.policy_value) < 1e-6
    assert never.client.calls == 0


def test_audit_is_recorded_for_every_decision(env, recs):
    econ, cfg = env
    b3 = B3Router(econ, cfg, client=MockLLMClient())
    evaluate(recs[:40], b3, econ=econ, cfg=cfg, bootstrap=0)
    assert b3.decisions
    for rec in b3.decisions.values():
        d = rec.to_dict()
        assert set(d) >= {"b2_proposal", "ambiguity", "llm_invoked",
                          "llm_proposal", "gates", "llm_accepted", "final"}
        if rec.llm_invoked:
            assert d["llm_proposal"] is not None and d["gates"]
