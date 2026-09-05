"""
Bounded transient retry, and bounded concurrency that cannot change results.

Two properties matter more than the mechanics:

  * A retry is the SAME decision attempt. It must not become a second billed
    call, a second decision node, or a second recovery -- otherwise retrying
    would inflate the very economics the benchmark exists to measure.

  * Retry must not soften fail-loud. A 503 is infrastructure noise and is worth
    another attempt; a 401, a malformed answer or a refusal is an ANSWER, and
    must fail on the first attempt exactly as before.
"""

from __future__ import annotations

import socket
import urllib.error

import pytest

from src import compliance, evaluate, generate, oracle
from src.policies import prewarm as pw
from src.policies.b3_router import B3Router
from src.policies.llm_client import (
    LLMClient, LLMDecision, MockLLMClient, observable_payload,
    payload_fingerprint,
)
from src.policies.retry import (
    MAX_ATTEMPTS, PERMANENT_STATUS, TRANSIENT_STATUS,
    backoff_delay, is_transient, status_of, with_retry,
)
from src.schema import ABANDON

NOSLEEP = lambda _s: None          # noqa: E731


def _flaky(*outcomes):
    """A callable that yields each outcome in turn; exceptions are raised."""
    seq = list(outcomes)
    def fn():
        o = seq.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o
    return fn


def _err(status):
    return urllib.error.HTTPError("u", status, f"status {status}", {}, None)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(TRANSIENT_STATUS))
def test_transient_statuses_are_retried(status):
    assert is_transient(_err(status)) is True


@pytest.mark.parametrize("status", sorted(PERMANENT_STATUS))
def test_permanent_statuses_are_never_retried(status):
    assert is_transient(_err(status)) is False


@pytest.mark.parametrize("exc", [
    socket.timeout("timed out"),
    TimeoutError("timed out"),
    ConnectionResetError("reset by peer"),
    ConnectionRefusedError("refused"),
    OSError("The read operation timed out"),
    urllib.error.URLError(socket.timeout("timed out")),
])
def test_connection_and_timeout_errors_are_retried(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("malformed model output"),
    KeyError("schema violation"),
    RuntimeError("model refused"),
    Exception("400 INVALID_ARGUMENT unsupported parameter"),
])
def test_answers_and_bad_requests_are_never_retried(exc):
    assert is_transient(exc) is False


def test_gemini_style_server_error_string_is_classified():
    """The live failure was `ServerError: 503 UNAVAILABLE ...`."""
    e = Exception("ServerError: 503 UNAVAILABLE. high demand, try again later")
    assert status_of(e) == 503 and is_transient(e) is True


# ---------------------------------------------------------------------------
# the retry loop
# ---------------------------------------------------------------------------

def test_503_then_success_recovers():
    fn = _flaky(_err(503), "ok")
    assert with_retry(fn, sleep=NOSLEEP) == "ok"


def test_429_then_success_recovers():
    fn = _flaky(_err(429), "ok")
    assert with_retry(fn, sleep=NOSLEEP) == "ok"


def test_timeout_then_success_recovers():
    fn = _flaky(socket.timeout("timed out"), "ok")
    assert with_retry(fn, sleep=NOSLEEP) == "ok"


def test_three_503s_exhaust_the_budget_and_raise():
    fn = _flaky(_err(503), _err(503), _err(503))
    with pytest.raises(urllib.error.HTTPError) as ei:
        with_retry(fn, sleep=NOSLEEP)
    assert ei.value.code == 503


def test_budget_is_bounded_to_three_attempts():
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _err(503)
    with pytest.raises(urllib.error.HTTPError):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == MAX_ATTEMPTS == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_failure_uses_only_one_attempt(status):
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _err(status)
    with pytest.raises(urllib.error.HTTPError):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == 1, "a 4xx must not consume the retry budget"


def test_malformed_output_is_not_retried():
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise ValueError("model returned invalid JSON")
    with pytest.raises(ValueError):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == 1


def test_backoff_is_exponential_bounded_and_jittered():
    import random
    r = random.Random(1)
    d1, d2, d3 = (backoff_delay(i, rng=r) for i in (1, 2, 3))
    assert d1 < d2 < d3                       # grows
    assert backoff_delay(50, rng=r) <= 8.0    # capped
    r2 = random.Random(2)
    assert backoff_delay(1, rng=r) != backoff_delay(1, rng=r2)  # jittered


def test_sleep_happens_between_attempts_not_after_the_last():
    slept = []
    fn = _flaky(_err(503), _err(503), _err(503))
    with pytest.raises(urllib.error.HTTPError):
        with_retry(fn, sleep=slept.append)
    assert len(slept) == MAX_ATTEMPTS - 1


# ---------------------------------------------------------------------------
# a retry is not a second decision
# ---------------------------------------------------------------------------

class _RetryOnceClient(LLMClient):
    """Fails transiently once per payload, then succeeds."""

    name = "retry-probe"

    def __init__(self):
        super().__init__()
        self.errors = 0
        self.retries = 0
        self.attempts = 3
        self.transport_calls = 0

    def _decide(self, payload):
        state = {"first": True}
        def _call():
            self.transport_calls += 1
            if state["first"]:
                state["first"] = False
                raise _err(503)
            return {"action": "ABANDON", "delay_hours": 0,
                    "confidence": 0.5, "rationale": "ok"}
        def _note(a, e, d):
            self.retries += 1
        from src.policies.llm_client import _coerce
        raw = with_retry(_call, attempts=self.attempts,
                         on_retry=_note, sleep=NOSLEEP)
        return _coerce(raw, "retry-probe")


def test_retry_does_not_create_extra_billed_calls_or_nodes():
    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = [r for r in generate.build(300, 7) if r.split == "test"][:25]

    c = _RetryOnceClient()
    router = B3Router(econ, cfg, client=c)
    m = evaluate.evaluate(recs, router, econ=econ, cfg=cfg,
                          bootstrap=0, collect_audit=True)

    routed_nodes = sum(1 for d in router.decisions.values() if d.llm_invoked)
    unique_payloads = len(c._cache)

    # Every transport call was retried once, so transport == 2x unique payloads
    assert c.transport_calls == 2 * unique_payloads
    assert c.retries == unique_payloads
    # ...but billed calls and decision nodes count the DECISION, not attempts.
    assert c.calls == unique_payloads
    assert routed_nodes <= len(router.decisions)
    # audit rows equal decision nodes, unaffected by retries
    audit_rows = sum(len(r.audit) for r in m.results)
    assert audit_rows == len(router.decisions)


def test_a_recovered_retry_is_not_counted_as_an_error():
    c = _RetryOnceClient()
    d = c.decide({"payment_id": "pay_1"})
    assert not d.malformed and c.errors == 0 and c.retries == 1


# ---------------------------------------------------------------------------
# concurrency preserves results and ordering
# ---------------------------------------------------------------------------

class _CountingMock(MockLLMClient):
    """Mock decisions, but with a call counter and an artificial reorder."""

    name = "counting-mock"

    def __init__(self):
        super().__init__()
        self.errors = 0
        self.retries = 0

    def _decide(self, payload):
        import time
        # Uneven latency so worker completion order differs from input order.
        time.sleep(0.002 * (hash(payload["payment_id"]) % 5))
        return super()._decide(payload)


def _run(client, recs, econ, cfg):
    router = B3Router(econ, cfg, client=client)
    m = evaluate.evaluate(recs, router, econ=econ, cfg=cfg,
                          bootstrap=0, collect_audit=True)
    rows = [(r.payment_id, s.step, s.chosen, s.chosen_delay)
            for r in m.results for s in r.audit]
    return m, rows


def test_concurrency_preserves_results_and_output_order():
    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = [r for r in generate.build(300, 7) if r.split == "test"][:40]

    seq_m, seq_rows = _run(_CountingMock(), recs, econ, cfg)

    conc = _CountingMock()
    def _discover(probe):
        _run(probe, recs, econ, cfg)
    summary = pw.prewarm(conc, _discover, concurrency=8)
    conc_m, conc_rows = _run(conc, recs, econ, cfg)

    assert summary["passes"] >= 1 and summary["decisions_fetched"] > 0
    assert conc_rows == seq_rows, "audit order or content changed under concurrency"
    assert conc_m.policy_value == pytest.approx(seq_m.policy_value)
    assert conc_m.regret == pytest.approx(seq_m.regret)


def test_prewarm_converges_and_leaves_no_misses():
    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = [r for r in generate.build(300, 7) if r.split == "test"][:30]
    c = _CountingMock()
    def _discover(probe):
        _run(probe, recs, econ, cfg)
    pw.prewarm(c, _discover, concurrency=4)
    probe = pw._DiscoveryClient(c)
    _run(probe, recs, econ, cfg)
    assert probe.misses == [], "a warm cache must produce zero misses"


def test_prewarm_bills_one_call_per_unique_payload():
    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = [r for r in generate.build(300, 7) if r.split == "test"][:30]
    c = _CountingMock()
    def _discover(probe):
        _run(probe, recs, econ, cfg)
    s = pw.prewarm(c, _discover, concurrency=4)
    assert c.calls == len(c._cache) == s["decisions_fetched"]


def test_discovery_client_never_calls_a_provider():
    real = _CountingMock()
    probe = pw._DiscoveryClient(real)
    with pytest.raises(AssertionError):
        probe._decide({"payment_id": "x"})
    d = probe.decide({"payment_id": "x"})
    assert d.malformed and probe.misses and real.calls == 0


def test_prewarm_rejects_nonsense_concurrency():
    with pytest.raises(ValueError):
        pw.prewarm(_CountingMock(), lambda p: None, concurrency=0)


# ---------------------------------------------------------------------------
# fail-loud survives all of this
# ---------------------------------------------------------------------------

def test_exhausted_retries_still_reject_the_run():
    from scripts.run_all import real_model_failure
    for backend in ("gemini", "nemotron", "anthropic"):
        assert real_model_failure(
            {"llm_errors": 1, "decision_nodes_routed": 700}, backend)


def test_no_backend_can_fall_back_to_mock():
    from src.policies.gemini_client import GeminiLLMClient
    from src.policies.openrouter_client import OpenRouterNemotronClient
    for cls in (GeminiLLMClient, OpenRouterNemotronClient):
        assert not issubclass(cls, MockLLMClient)
        assert cls.name != MockLLMClient.name


def test_prewarm_reconciles_error_count_against_decisions():
    """Worker threads race on `errors += 1`; the count must not undercount."""
    class _AllFail(LLMClient):
        name = "all-fail"
        def __init__(self):
            super().__init__()
            self.errors = 0
            self.retries = 0
        def _decide(self, payload):
            return LLMDecision(ABANDON, 0, 0.0, "boom", "x", malformed=True)

    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = [r for r in generate.build(300, 7) if r.split == "test"][:10]
    c = _AllFail()
    def _discover(probe):
        _run(probe, recs, econ, cfg)
    s = pw.prewarm(c, _discover, concurrency=4)
    assert c.errors >= 1 and s["errors"] >= 1
    from scripts.run_all import real_model_failure
    assert real_model_failure({"llm_errors": c.errors,
                               "decision_nodes_routed": 10}, "gemini")


def test_adapters_expose_the_retry_budget_and_counter():
    from src.policies.gemini_client import GeminiLLMClient
    from src.policies.openrouter_client import OpenRouterNemotronClient
    c = OpenRouterNemotronClient(api_key="not-real")
    assert c.attempts == MAX_ATTEMPTS and c.retries == 0
    assert "attempts" in GeminiLLMClient.__init__.__code__.co_varnames


# ---------------------------------------------------------------------------
# A 429 means two different things
# ---------------------------------------------------------------------------
#
# The 100-record dry run returned `429 RESOURCE_EXHAUSTED - you exceeded your
# current quota` on 140 of 209 calls, and the retry logic turned those into 311
# further attempts that could not possibly have succeeded. A rate-limit 429 is
# worth backing off from; an exhausted-quota 429 is terminal.

from src.policies.retry import QUOTA_MARKERS, is_quota_exhausted


@pytest.mark.parametrize("msg", [
    "429 Too Many Requests",
    "429 rate limit reached, please slow down",
    "HTTP 429: too many requests, retry after 2s",
])
def test_ordinary_rate_limit_429_still_retries(msg):
    assert is_transient(Exception(msg)) is True
    assert is_quota_exhausted(Exception(msg)) is False


@pytest.mark.parametrize("msg", [
    "ClientError: 429 RESOURCE_EXHAUSTED.",
    "429: You exceeded your current quota, please check your plan and billing details",
    "429 quota exceeded for this project",
    "429 out of quota",
    "429 insufficient_quota",
    # the verbatim shape seen in the live dry run
    "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota, please check your plan and billing details.'}}",
])
def test_quota_exhausted_429_does_not_retry(msg):
    exc = Exception(msg)
    assert is_quota_exhausted(exc) is True
    assert is_transient(exc) is False, "an exhausted quota must fail fast"


def test_quota_429_consumes_only_one_attempt():
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise Exception("429 RESOURCE_EXHAUSTED: you exceeded your current quota")
    with pytest.raises(Exception):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == 1, "quota exhaustion must not burn the retry budget"


def test_rate_limit_429_still_uses_the_full_budget():
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise Exception("429 Too Many Requests")
    with pytest.raises(Exception):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == MAX_ATTEMPTS == 3


def test_quota_markers_are_matched_case_insensitively():
    for marker in QUOTA_MARKERS:
        assert is_quota_exhausted(Exception(f"429 {marker.upper()}"))


def test_503_is_unaffected_by_the_quota_rule():
    """A 503 stays retryable even if the body happens to mention quota."""
    assert is_transient(Exception("503 UNAVAILABLE high demand")) is True
    assert is_transient(Exception("503 UNAVAILABLE quota exceeded upstream")) is True


def test_transport_failures_are_unaffected_by_the_quota_rule():
    for exc in (socket.timeout("timed out"), ConnectionResetError("reset"),
                ConnectionRefusedError("refused"), TimeoutError("timed out")):
        assert is_transient(exc) is True


def test_malformed_output_still_does_not_retry_after_the_change():
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise ValueError("model returned invalid JSON")
    with pytest.raises(ValueError):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 422])
def test_permanent_4xx_still_does_not_retry_after_the_change(status):
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _err(status)
    with pytest.raises(urllib.error.HTTPError):
        with_retry(fn, sleep=NOSLEEP)
    assert calls["n"] == 1
