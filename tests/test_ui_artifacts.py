"""
Guards for the two generated presentation artifacts: the control-center page
and the README charts.

These are the parts a Buildathon judge actually looks at, and both are derived
from the benchmark rather than authored by hand. Three things must stay true:

  1. The page never shows the benchmark's answer key. A production recovery
     system does not know the true failure cause or the oracle-optimal action,
     so a demo that displays them is showing something the product could not
     have. tests/test_no_leakage.py guards the decision path; this guards the
     presentation layer, which is a separate hole.

  2. Both artifacts are pure functions of results/metrics.json. If a number in
     a chart or on a KPI card can drift away from the benchmark, the demo stops
     being evidence.

  3. The SVG is well-formed. The first draft of make_charts.py interpolated a
     font stack containing double quotes into a double-quoted SVG attribute,
     which produced a file that looked fine to Python and failed to parse in
     every renderer. A syntax check is cheap; discovering it in front of a
     judge is not.
"""

from __future__ import annotations

import dataclasses
import json
import re
import xml.dom.minidom

import pytest

from scripts import build_ui, make_charts
from src.schema import HiddenState

HIDDEN_FIELDS = {f.name for f in dataclasses.fields(HiddenState)}


@pytest.fixture(scope="module")
def payload():
    return build_ui.load_payload()


@pytest.fixture(scope="module")
def page(payload):
    return build_ui.build(payload)


@pytest.fixture(scope="module")
def metrics(payload):
    return payload["metrics"]


# ---------------------------------------------------------------------------
# 1. The answer key must not reach the product UI
# ---------------------------------------------------------------------------

def test_payload_contains_no_hidden_field_names(payload):
    blob = json.dumps(payload)
    leaked = sorted(n for n in HIDDEN_FIELDS if f'"{n}"' in blob)
    assert not leaked, f"answer key in UI payload: {leaked}"


def test_rendered_page_contains_no_hidden_field_names(page):
    leaked = sorted(n for n in HIDDEN_FIELDS if f'"{n}"' in page)
    assert not leaked, f"answer key in rendered page: {leaked}"


def test_payment_records_expose_only_allowlisted_observables(payload):
    for pid, rec in payload["payments"].items():
        extra = set(rec) - set(build_ui.DETAIL_FIELDS) - {"slice_tag"}
        assert not extra, f"{pid} carries unlisted fields: {extra}"
        assert not set(rec) & HIDDEN_FIELDS


def test_hidden_state_checker_actually_rejects_a_leak():
    """The guard must bite, not pass vacuously."""
    with pytest.raises(AssertionError):
        build_ui.check_no_hidden_state({"oops": {"true_reason": "MANDATE_REVOKED"}})


# ---------------------------------------------------------------------------
# 2. Derived from the benchmark, not authored
# ---------------------------------------------------------------------------

def test_page_is_deterministic(payload):
    assert build_ui.build(payload) == build_ui.build(payload)


def test_audit_row_count_matches_the_persisted_trail(payload):
    with open("results/b3_audit.jsonl") as f:
        n = sum(1 for line in f if line.strip())
    assert len(payload["audit"]) == n


def test_population_matches_the_frozen_split(payload, metrics):
    assert len(payload["payments"]) == metrics["n"]
    assert metrics["split"] == "test"


def test_every_audited_payment_has_observables(payload):
    missing = {a["payment_id"] for a in payload["audit"]} - set(payload["payments"])
    assert not missing, f"audit references payments the UI cannot describe: {missing}"


def test_page_states_the_real_headline_numbers(page, metrics):
    e = metrics["b3_economics"]
    b2 = next(p for p in metrics["policies"] if p["policy"] == "B2 rules")
    # The population size is stated in plain language now, not as "n=564":
    # internal identifiers (file paths, split names, seeds) were removed from
    # the page on purpose, so this asserts the FIGURE is present rather than
    # any particular technical spelling of it.
    assert str(metrics["n"]) in page
    assert f"{100 * b2['share_of_oracle']:.1f}" in page
    assert f"{abs(e['net_benefit_vs_b2']):,.0f}" in page


def test_mock_backend_is_disclosed_on_the_page(page, metrics):
    """A judge must not be able to mistake MockLLM for a real model run."""
    if metrics["llm_backend"] == "mock":
        assert "MockLLM" in page
        assert "no real model benchmarked" in page


def test_negative_result_is_not_hidden(page, metrics):
    e = metrics["b3_economics"]
    if e["net_benefit_vs_b2"] < 0:
        assert "did not earn its cost" in page
        assert "More AI" in page


# ---------------------------------------------------------------------------
# 3. Charts: well-formed, and faithful to metrics.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def charts(metrics):
    return {"ladder": make_charts.policy_ladder(metrics),
            "economics": make_charts.ai_economics(metrics)}


@pytest.mark.parametrize("name", ["ladder", "economics"])
def test_chart_is_well_formed_xml(charts, name):
    xml.dom.minidom.parseString(charts[name])


@pytest.mark.parametrize("name", ["ladder", "economics"])
def test_chart_attributes_are_not_broken_by_quotes(charts, name):
    """Regression: a double-quoted font family used to terminate the attribute."""
    for attr in re.findall(r'font-family="([^"]*)"', charts[name]):
        assert '"' not in attr


def test_ladder_chart_uses_an_honest_full_axis(charts):
    svg = charts["ladder"]
    for tick in ("0%", "25%", "50%", "75%", "100%"):
        assert f">{tick}<" in svg


def test_ladder_chart_values_come_from_metrics(charts, metrics):
    svg = charts["ladder"]
    for p in metrics["policies"]:
        assert f"{100 * p['share_of_oracle']:.1f}%" in svg


def test_economics_chart_shows_the_negative_net(charts, metrics):
    svg = charts["economics"]
    net = metrics["b3_economics"]["net_benefit_vs_b2"]
    assert net < 0, "fixture assumes the benchmark's negative result"
    assert f"−₹{abs(net):,.0f}" in svg
    assert "more AI" in svg and "more revenue" in svg


def test_economics_chart_marks_a_zero_baseline(charts):
    """A negative bar is meaningless without a visible zero."""
    assert ">0<" in charts["economics"]
