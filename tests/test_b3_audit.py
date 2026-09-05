"""
Audit-trail integrity and LLM cost accounting.

Two things were previously asserted only by inspection:

  1. That the `final_action` B3 records is the action the evaluator actually
     executed. B3 returns a RANKED list and the harness walks it taking the
     first entry that clears compliance, so a divergence is structurally
     possible: the audit could name the LLM's action while the harness ran
     B2's. That would make every routed record's trail untrustworthy.

  2. That reach-weighted LLM cost is arithmetically right. A decision at step k
     only happens if every earlier attempt failed, so invocations are weighted
     by the product of (1 - p_success) over strictly earlier steps. An
     off-by-one there -- multiplying reach before counting the invocation
     rather than after -- silently misprices the entire B3 economics argument.

Both are checked against the frozen simulator with no sampling anywhere.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src import compliance, evaluate, generate, oracle
from src.policies.b2_rules import B2Rules
from src.policies.b3_router import B3Decision, B3Router
from src.policies.llm_client import LLMClient, LLMDecision, MockLLMClient
from src.schema import (
    Observation, HiddenState, Record,
    ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
    GATEWAY_TIMEOUT,
)
from scripts.run_all import (llm_accounting, real_model_failure,
                            write_b3_audit)


@pytest.fixture(scope="module")
def env():
    return oracle.load_economics(), compliance.load_config()


@pytest.fixture(scope="module")
def recs():
    return [r for r in generate.build(600, 7) if r.split == "test"]


# ---------------------------------------------------------------------------
# 3. Audit integrity: persisted final_action == what the evaluator ran
# ---------------------------------------------------------------------------

def _persist(env, recs, router):
    econ, cfg = env
    m = evaluate.evaluate(recs, router, econ=econ, cfg=cfg,
                          bootstrap=0, collect_audit=True)
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        write_b3_audit(path, m, router)
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    finally:
        os.unlink(path)
    return m, rows


def test_persisted_final_action_matches_executed_action(env, recs):
    """The headline integrity guarantee, over the real mixed population."""
    m, rows = _persist(env, recs, B3Router(*env, client=MockLLMClient()))
    assert rows, "no audit rows written"
    for row in rows:
        assert row["final"]["action"] == row["executed"]["action"], (
            f"{row['payment_id']} step {row['step']}: audit says "
            f"{row['final']['action']} but evaluator ran "
            f"{row['executed']['action']}"
        )
        assert row["final"]["delay_hours"] == row["executed"]["delay_hours"]


def test_integrity_holds_on_accepted_llm_proposals(env, recs):
    """Cover the accepted-LLM path specifically, not just in aggregate."""
    m, rows = _persist(env, recs, B3Router(*env, client=MockLLMClient()))
    accepted = [r for r in rows if r["llm_invoked"] and r["llm_accepted"]]
    assert accepted, "no accepted LLM proposals in this population"
    for row in accepted:
        assert row["final"]["action"] == row["llm_proposal"]["action"]
        assert row["final"]["action"] == row["executed"]["action"]


def test_integrity_holds_on_rejected_llm_proposals(env, recs):
    """Rejected proposals must execute B2's action, not the LLM's."""
    econ, cfg = env
    router = B3Router(econ, cfg, client=MockLLMClient())
    m, rows = _persist(env, recs, router)
    rejected = [r for r in rows if r["llm_invoked"] and not r["llm_accepted"]]
    assert rejected, "no rejected LLM proposals in this population"
    for row in rejected:
        assert row["final"]["action"] == row["b2_proposal"]["action"]
        assert row["final"]["action"] == row["executed"]["action"]
        assert any(v.startswith("rejected") for v in row["gates"].values())


def test_integrity_holds_when_llm_is_never_consulted(env, recs):
    """The non-routed majority must still be faithfully recorded."""
    econ, cfg = env
    m, rows = _persist(env, recs,
                       B3Router(econ, cfg, client=MockLLMClient(),
                                ambiguity_threshold=-1.0))
    assert rows and all(not r["llm_invoked"] for r in rows)
    for row in rows:
        assert row["final"]["action"] == row["b2_proposal"]["action"]
        assert row["final"]["action"] == row["executed"]["action"]


def test_audit_row_carries_the_required_fields(env, recs):
    m, rows = _persist(env, recs, B3Router(*env, client=MockLLMClient()))
    required = {"payment_id", "b2_proposal", "ambiguity", "llm_invoked",
                "llm_proposal", "gates", "llm_accepted", "final", "executed",
                "p_success", "reach", "step", "slice_tag"}
    for row in rows:
        assert required <= set(row), f"missing {required - set(row)}"
    routed = [r for r in rows if r["llm_invoked"]]
    assert routed
    for row in routed:
        assert row["llm_proposal"]["confidence"] is not None
        assert row["gates"]


def test_reach_is_a_probability_and_decreases_along_an_episode(env, recs):
    m, rows = _persist(env, recs, B3Router(*env, client=MockLLMClient()))
    by_pid = {}
    for row in rows:
        by_pid.setdefault(row["payment_id"], []).append(row)
    for pid, steps in by_pid.items():
        steps.sort(key=lambda r: r["step"])
        assert steps[0]["reach"] == 1.0, f"{pid} first step reach != 1"
        prev = 1.0
        for s in steps:
            assert 0.0 <= s["reach"] <= 1.0
            assert s["reach"] <= prev + 1e-9
            prev = s["reach"]


# ---------------------------------------------------------------------------
# 4. Cost accounting: hand-computable two-step episode
# ---------------------------------------------------------------------------

class _TwoStepRouter:
    """
    Deterministic stand-in exposing exactly the surface llm_accounting reads.

    Retries twice at a fixed 6h delay, then stops, and marks both retry
    decisions as LLM-invoked. Using a real B3Router here would make the
    invocation pattern depend on B2's ranking, which is the opposite of
    hand-computable.
    """

    NAME = "B3 router"

    def __init__(self):
        self.decisions = {}
        self.client = MockLLMClient()

    def propose(self, obs):
        stop = obs.attempts_used >= 2
        action, delay = (ABANDON, 0) if stop else (RETRY_LATER, 6)
        self.decisions[(obs.payment_id, obs.elapsed_hours)] = B3Decision(
            payment_id=obs.payment_id, elapsed_hours=obs.elapsed_hours,
            attempts_used=obs.attempts_used,
            b2_action=action, b2_delay=delay, b2_ev=0.0,
            margin=0.0, relative_margin=0.0, stake=0.0,
            ambiguous=not stop, llm_invoked=not stop,
            llm=None if stop else LLMDecision(action, delay, 0.9, "probe"),
            llm_accepted=not stop,
            final_action=action, final_delay=delay,
        )
        return [(action, delay)]


def _gateway_timeout_record():
    """
    One record whose success probabilities are trivial to compute by hand.

    simulator.success_probability for GATEWAY_TIMEOUT is 0.88 once landing >= 1,
    scaled by _staleness (1.0 within 72h) and _attempt_fatigue (0.97 ** attempts).
    """
    obs = Observation(
        payment_id="pay_probe", amount=1000.0, currency="INR",
        decline_code=GATEWAY_TIMEOUT, payment_method="card",
        customer_tenure_days=400, prior_successes=12, prior_failures=0,
        days_into_billing_cycle=1, day_of_month=10, hour_of_day=11,
        mandate_status="active", mandate_cap=5000.0, recent_refund=False,
        active_dispute=False, pre_debit_notice_sent=True,
        attempts_already_made=0, subscription_plan_value=1000.0,
    )
    return Record(obs=obs, hidden=HiddenState(true_reason=GATEWAY_TIMEOUT),
                  slice_tag="ordinary", split="test")


def test_reach_weighted_llm_cost_matches_hand_computation(env):
    econ, cfg = env
    rec = _gateway_timeout_record()
    router = _TwoStepRouter()

    b3 = evaluate.evaluate([rec], router, name="B3 router", econ=econ, cfg=cfg,
                           bootstrap=0, collect_audit=True)
    b2 = evaluate.evaluate([rec], B2Rules(econ, cfg), econ=econ, cfg=cfg,
                           bootstrap=0)
    acct = llm_accounting(b3, router, b2, econ)

    # Hand computation, from the frozen simulator:
    #   step 0: elapsed 0, delay 6  -> landing 6,  attempts_used 0
    #           p0 = 0.88 * staleness(6)=1.0 * 0.97**0 = 0.88
    #   step 1: elapsed 7, delay 6  -> landing 13, attempts_used 1
    #           p1 = 0.88 * staleness(13)=1.0 * 0.97**1 = 0.8536
    #   step 2: attempts_used 2 -> ABANDON, not invoked
    p0, p1 = 0.88, 0.88 * 0.97
    steps = b3.results[0].audit
    assert len(steps) == 3, [s.chosen for s in steps]
    assert steps[0].p_success == pytest.approx(p0)
    assert steps[1].p_success == pytest.approx(p1)
    assert steps[2].chosen == ABANDON

    # reach(step 0) = 1; reach(step 1) = 1 - p0. Step 2 is not invoked.
    expected_invocations = 1.0 + (1.0 - p0)
    assert acct["decision_nodes_routed"] == 2
    assert acct["expected_invocations"] == pytest.approx(expected_invocations)
    assert acct["expected_invocations"] == pytest.approx(1.12)
    assert acct["llm_cost"] == pytest.approx(1.12 * econ["costs"]["llm_decision"])
    assert acct["llm_cost"] == pytest.approx(0.392)


def test_reach_weighting_is_strictly_cheaper_than_naive_node_count(env):
    """
    The whole point of reach weighting: deep retries usually never happen, so
    billing every node at full price overstates the cost.
    """
    econ, cfg = env
    rec = _gateway_timeout_record()
    router = _TwoStepRouter()
    b3 = evaluate.evaluate([rec], router, name="B3 router", econ=econ, cfg=cfg,
                           bootstrap=0, collect_audit=True)
    b2 = evaluate.evaluate([rec], B2Rules(econ, cfg), econ=econ, cfg=cfg,
                           bootstrap=0)
    acct = llm_accounting(b3, router, b2, econ)
    assert acct["expected_invocations"] < acct["decision_nodes_routed"]


def test_cost_accounting_is_deterministic(env):
    econ, cfg = env
    rec = _gateway_timeout_record()
    runs = []
    for _ in range(3):
        router = _TwoStepRouter()
        b3 = evaluate.evaluate([rec], router, name="B3 router", econ=econ,
                               cfg=cfg, bootstrap=0, collect_audit=True)
        b2 = evaluate.evaluate([rec], B2Rules(econ, cfg), econ=econ, cfg=cfg,
                               bootstrap=0)
        runs.append(llm_accounting(b3, router, b2, econ))
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# Real-model run must fail loudly, not silently become the rules baseline
# ---------------------------------------------------------------------------

class _ErroringClient(LLMClient):
    """Stands in for AnthropicLLMClient after the API has failed."""

    name = "anthropic"

    def __init__(self, errors=0):
        super().__init__()
        self.errors = errors

    def _decide(self, payload):
        return LLMDecision(ABANDON, 0, 0.0, "api error", "anthropic",
                           malformed=True)


def test_accounting_surfaces_llm_error_count(env):
    """The count must reach the report; it was previously incremented and dropped."""
    econ, cfg = env
    rec = _gateway_timeout_record()
    router = _TwoStepRouter()
    router.client = _ErroringClient(errors=7)
    b3 = evaluate.evaluate([rec], router, name="B3 router", econ=econ, cfg=cfg,
                           bootstrap=0, collect_audit=True)
    b2 = evaluate.evaluate([rec], B2Rules(econ, cfg), econ=econ, cfg=cfg,
                           bootstrap=0)
    assert llm_accounting(b3, router, b2, econ)["llm_errors"] == 7


def test_accounting_defaults_to_zero_for_a_client_without_errors(env):
    """MockLLMClient has no .errors attribute; the lookup must not blow up."""
    econ, cfg = env
    rec = _gateway_timeout_record()
    router = _TwoStepRouter()
    assert not hasattr(router.client, "errors")
    b3 = evaluate.evaluate([rec], router, name="B3 router", econ=econ, cfg=cfg,
                           bootstrap=0, collect_audit=True)
    b2 = evaluate.evaluate([rec], B2Rules(econ, cfg), econ=econ, cfg=cfg,
                           bootstrap=0)
    assert llm_accounting(b3, router, b2, econ)["llm_errors"] == 0


def test_real_model_run_is_rejected_when_any_call_failed():
    """
    One failed call is enough. A partially-failed run is a blend of the model
    and the rules baseline, and there is no honest way to label that.
    """
    msg = real_model_failure({"llm_errors": 1, "decision_nodes_routed": 707},
                             "anthropic")
    assert msg
    assert "1 of 707" in msg
    assert "NOT a" in msg and "NOT written" in msg


def test_real_model_run_is_accepted_when_no_call_failed():
    assert real_model_failure({"llm_errors": 0, "decision_nodes_routed": 707},
                              "anthropic") == ""


def test_mock_backend_is_never_gated_by_the_error_check():
    """The mock cannot error; gating it would break the offline path."""
    assert real_model_failure({"llm_errors": 0, "decision_nodes_routed": 707},
                              "mock") == ""
    assert real_model_failure({"llm_errors": 99, "decision_nodes_routed": 707},
                              "mock") == ""


def test_failure_check_tolerates_a_missing_error_key():
    assert real_model_failure({"decision_nodes_routed": 707}, "anthropic") == ""


# ---------------------------------------------------------------------------
# A second real backend must not be able to diverge from the first
# ---------------------------------------------------------------------------

def test_gemini_backend_shares_the_prompt_and_schema():
    """
    A per-provider prompt would quietly turn a model comparison into a prompt
    comparison, so both real backends must be held to the identical contract.
    """
    from src.policies import gemini_client as g
    from src.policies import llm_client as base

    assert issubclass(g.GeminiLLMClient, base.LLMClient)
    assert g.SYSTEM_PROMPT is base.SYSTEM_PROMPT
    assert set(g._GEMINI_SCHEMA["properties"]) == set(base.DECISION_SCHEMA["properties"])
    assert g._GEMINI_SCHEMA["required"] == base.DECISION_SCHEMA["required"]
    assert g._GEMINI_SCHEMA["properties"]["action"]["enum"] == list(base.VALID_ACTIONS)


def test_every_non_mock_backend_is_gated_by_the_failure_check():
    """
    The fail-loud guard keys off the backend name. Adding a provider must not
    create a path where errors are silently tolerated.
    """
    from src.policies import gemini_client as g
    from src.policies.llm_client import AnthropicLLMClient, MockLLMClient

    for name in (g.GeminiLLMClient.name, AnthropicLLMClient.name):
        assert name != MockLLMClient.name
        assert real_model_failure({"llm_errors": 1, "decision_nodes_routed": 10}, name)
    assert real_model_failure({"llm_errors": 1, "decision_nodes_routed": 10},
                              MockLLMClient.name) == ""


def test_gemini_client_needs_no_key_to_import():
    """The module must be importable in a clone with no SDK and no credentials."""
    from src.policies.gemini_client import DEFAULT_MODEL, GeminiLLMClient
    assert DEFAULT_MODEL and GeminiLLMClient.name == "gemini"


# ---------------------------------------------------------------------------
# OpenRouter / Nemotron provider adapter
# ---------------------------------------------------------------------------

def test_nemotron_missing_api_key_fails_loudly(monkeypatch):
    """
    A keyless run must stop before evaluation starts. Falling back to the mock
    would publish a rules-only result under a model's name.
    """
    from src.policies.openrouter_client import MissingAPIKey, OpenRouterNemotronClient
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey) as ei:
        OpenRouterNemotronClient()
    msg = str(ei.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "never falls back" in msg


def test_nemotron_reuses_the_exact_frozen_prompt_and_schema():
    """The schema is installed as the tool's parameters -- not a copy."""
    from src.policies import openrouter_client as o
    from src.policies import llm_client as base

    assert issubclass(o.OpenRouterNemotronClient, base.LLMClient)
    assert o.SYSTEM_PROMPT is base.SYSTEM_PROMPT
    assert o.DECISION_TOOL["function"]["parameters"] is base.DECISION_SCHEMA
    assert o.DEFAULT_MODEL == "nvidia/nemotron-3.5-lightning:free"


def test_nemotron_does_not_send_unsupported_response_format(monkeypatch):
    """
    The model's supported_parameters do not include response_format, and
    OpenRouter fails a request carrying an unsupported parameter. Assert the
    body uses forced tool calling and only supported knobs.
    """
    from src.policies.openrouter_client import OpenRouterNemotronClient
    sent = {}
    c = OpenRouterNemotronClient(api_key="test-key-not-real")
    monkeypatch.setattr(c, "_post", lambda body: sent.update(body) or {
        "choices": [{"message": {"tool_calls": [{"function": {
            "arguments": '{"action":"ABANDON","delay_hours":0,'
                         '"confidence":0.5,"rationale":"x"}'}}]}}]})
    c._decide({"payment_id": "pay_1"})
    assert "response_format" not in sent
    assert sent["tool_choice"]["function"]["name"] == "record_recovery_decision"

    # Every tunable key in the body must be one this model actually accepts.
    # Read from the live OpenRouter models API for
    # nvidia/nemotron-3.5-lightning:free, which reports exactly:
    #   include_reasoning, max_tokens, reasoning, seed, temperature,
    #   tool_choice, tools, top_p
    # OpenRouter fails a request carrying an unsupported parameter, so this is
    # the invariant that matters -- not a hand-kept allow-list.
    SUPPORTED = {"include_reasoning", "max_tokens", "reasoning", "seed",
                 "temperature", "tool_choice", "tools", "top_p"}
    STRUCTURAL = {"model", "messages"}          # not tunable parameters
    unexpected = set(sent) - SUPPORTED - STRUCTURAL
    assert not unexpected, f"body carries unsupported parameters: {unexpected}"

    # Reasoning is deliberately disabled: this model spends its whole budget
    # thinking otherwise (292 reasoning tokens on a trivial probe, 9-167s per
    # decision, sometimes finish_reason=length before answering).
    assert sent["reasoning"] == {"enabled": False}


@pytest.mark.parametrize("resp", [
    {},                                                    # empty body
    {"choices": []},                                       # no choice
    {"choices": [{"message": {}}]},                        # no tool call
    {"choices": [{"message": {"content": "I cannot help"}}]},   # prose refusal
    {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "{oops"}}]}}]},
])
def test_nemotron_malformed_output_is_counted_and_rejected(resp, monkeypatch):
    from src.policies.openrouter_client import OpenRouterNemotronClient
    c = OpenRouterNemotronClient(api_key="test-key-not-real")
    monkeypatch.setattr(c, "_post", lambda body: resp)
    d = c._decide({"payment_id": "pay_1"})
    assert d.malformed and c.errors == 1
    assert real_model_failure({"llm_errors": c.errors, "decision_nodes_routed": 1},
                              c.name)


def test_nemotron_api_error_is_counted_not_swallowed(monkeypatch):
    from src.policies.openrouter_client import OpenRouterNemotronClient
    c = OpenRouterNemotronClient(api_key="test-key-not-real")

    def boom(body):
        raise ConnectionResetError("connection reset")
    monkeypatch.setattr(c, "_post", boom)
    d = c._decide({"payment_id": "pay_1"})
    assert d.malformed and c.errors == 1
    assert "ConnectionResetError" in d.rationale


def test_nemotron_accepts_arguments_as_string_or_object(monkeypatch):
    """OpenRouter's docs do not state which; both must work."""
    from src.policies.openrouter_client import OpenRouterNemotronClient
    obj = {"action": "RETRY_LATER", "delay_hours": 24,
           "confidence": 0.8, "rationale": "funds likely to land"}
    for args in (obj, json.dumps(obj)):
        c = OpenRouterNemotronClient(api_key="test-key-not-real")
        monkeypatch.setattr(c, "_post", lambda body: {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": args}}]}}]})
        d = c._decide({"payment_id": "pay_1"})
        assert not d.malformed and d.action == "RETRY_LATER" and d.delay_hours == 24
        assert c.errors == 0


def test_nemotron_output_still_goes_through_the_frozen_validation(monkeypatch):
    """A confident but invalid action must be rejected by _coerce, not trusted."""
    from src.policies.openrouter_client import OpenRouterNemotronClient
    c = OpenRouterNemotronClient(api_key="test-key-not-real")
    monkeypatch.setattr(c, "_post", lambda body: {
        "choices": [{"message": {"tool_calls": [{"function": {"arguments":
            '{"action":"REFUND_EVERYTHING","delay_hours":0,'
            '"confidence":0.99,"rationale":"trust me"}'}}]}}]})
    d = c._decide({"payment_id": "pay_1"})
    assert d.malformed and d.action == "ABANDON"


def test_nemotron_never_silently_becomes_the_mock():
    from src.policies.openrouter_client import OpenRouterNemotronClient
    from src.policies.llm_client import MockLLMClient
    assert OpenRouterNemotronClient.name == "nemotron" != MockLLMClient.name
    assert real_model_failure({"llm_errors": 1, "decision_nodes_routed": 5},
                              OpenRouterNemotronClient.name)
    assert not issubclass(OpenRouterNemotronClient, MockLLMClient)


def test_api_key_never_appears_in_the_module_or_an_error(monkeypatch):
    """The key lives only in a request header, never in output."""
    from src.policies.openrouter_client import OpenRouterNemotronClient
    import src.policies.openrouter_client as mod
    src = open(mod.__file__).read()
    # The env var NAME appears in the help text ("export OPENROUTER_API_KEY=...")
    # which is correct and not a leak. What must not appear is a key-shaped
    # literal, or an assignment of one.
    import re
    assert not re.search(r'sk-or-v1-[A-Za-z0-9_-]{8,}', src)
    assert not re.search(r'_key\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', src)

    secret = "sk-or-v1-DO-NOT-LEAK-ME"
    c = OpenRouterNemotronClient(api_key=secret)

    def boom(body):
        raise RuntimeError("upstream exploded")
    monkeypatch.setattr(c, "_post", boom)
    d = c._decide({"payment_id": "pay_1"})
    assert secret not in d.rationale and secret not in repr(d)
