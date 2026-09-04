"""
Enforces the observable / hidden separation.

src/schema.py claims "if a policy ever touches HiddenState, the benchmark is
void. tests/test_no_leakage.py enforces this." This is that file. It checks the
claim structurally rather than trusting reviewers to notice.

Three layers:
  1. The two dataclasses cannot share a field.
  2. Only whitelisted modules may import HiddenState at all.
  3. Nothing in src/policies/ may reach for ground truth, by any spelling.
"""

from __future__ import annotations

import ast
import dataclasses
import os

import pytest

from src.schema import Observation, HiddenState, Record

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
POLICIES = os.path.join(SRC, "policies")

# Field names that exist only as ground truth. A policy seeing any of these
# means the score is meaningless.
HIDDEN_FIELDS = {
    "true_reason", "funds_return_hours", "outage_duration_hours",
    "update_responsiveness", "optimal_action", "optimal_delay_hours",
    "oracle_ev", "truly_recoverable",
}

# The only modules with a legitimate reason to know ground truth: the world
# model, the upper-bound solver, the generator that builds it, and the schema
# that declares it.
MAY_SEE_HIDDEN = {"schema", "simulator", "oracle", "generate"}


def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


def _py_files(root: str):
    for dirpath, _, names in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n.endswith(".py"):
                yield os.path.join(dirpath, n)


def _module_name(path: str) -> str:
    return os.path.splitext(os.path.relpath(path, SRC))[0].replace(os.sep, ".")


# ---------------------------------------------------------------------------
# 1. The dataclasses are disjoint
# ---------------------------------------------------------------------------

def test_observation_and_hidden_share_no_fields():
    overlap = _fields(Observation) & _fields(HiddenState)
    assert not overlap, f"fields exist in both Observation and HiddenState: {overlap}"


def test_observation_declares_no_hidden_field():
    leaked = _fields(Observation) & HIDDEN_FIELDS
    assert not leaked, f"Observation exposes ground truth: {leaked}"


def test_hidden_fields_constant_matches_schema():
    """If HiddenState grows a field, this test must be updated deliberately."""
    unlisted = _fields(HiddenState) - HIDDEN_FIELDS
    assert not unlisted, (
        f"HiddenState has fields not listed in HIDDEN_FIELDS: {unlisted}. "
        "Add them there so the policy guard covers them."
    )


def test_serialised_observation_carries_no_ground_truth():
    obs_keys = set(Observation(
        payment_id="pay_00000", amount=499.0, currency="INR",
        decline_code="DO_NOT_HONOUR", payment_method="card",
        customer_tenure_days=100, prior_successes=3, prior_failures=1,
        days_into_billing_cycle=0, day_of_month=12, hour_of_day=10,
        mandate_status="active", mandate_cap=999.0, recent_refund=False,
        active_dispute=False, pre_debit_notice_sent=True,
        attempts_already_made=0, subscription_plan_value=499.0,
    ).to_dict())
    assert not obs_keys & HIDDEN_FIELDS


def test_record_keeps_layers_in_separate_branches():
    """A record's obs branch must never carry hidden keys after a round-trip."""
    d = Record(
        obs=Observation(
            payment_id="pay_00001", amount=99.0, currency="INR",
            decline_code="INSUFFICIENT_FUNDS", payment_method="upi_autopay",
            customer_tenure_days=30, prior_successes=1, prior_failures=0,
            days_into_billing_cycle=1, day_of_month=3, hour_of_day=9,
            mandate_status="active", mandate_cap=200.0, recent_refund=False,
            active_dispute=False, pre_debit_notice_sent=True,
            attempts_already_made=0, subscription_plan_value=99.0,
        ),
        hidden=HiddenState(true_reason="INSUFFICIENT_FUNDS", funds_return_hours=20),
    ).to_dict()
    assert not set(d["obs"]) & HIDDEN_FIELDS
    assert "true_reason" in d["hidden"], "hidden branch should still hold truth"


# ---------------------------------------------------------------------------
# 2. Import whitelist
# ---------------------------------------------------------------------------

def _imports_hidden_state(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "HiddenState" for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name.endswith("HiddenState") for a in node.names):
                return True
    return False


@pytest.mark.parametrize("path", sorted(_py_files(SRC)))
def test_only_whitelisted_modules_import_hidden_state(path):
    mod = _module_name(path)
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    if _imports_hidden_state(tree):
        top = mod.split(".")[0]
        assert top in MAY_SEE_HIDDEN, (
            f"{mod} imports HiddenState but is not in MAY_SEE_HIDDEN. "
            "Either it should not see ground truth, or the whitelist needs a "
            "deliberate, reviewed change."
        )


# ---------------------------------------------------------------------------
# 3. Policies cannot reach for truth by any spelling
# ---------------------------------------------------------------------------

def _policy_files():
    return sorted(_py_files(POLICIES)) if os.path.isdir(POLICIES) else []


@pytest.mark.parametrize("path", _policy_files() or [None])
def test_policies_never_touch_ground_truth(path):
    if path is None:
        pytest.skip("no policy modules yet")

    with open(path) as f:
        src = f.read()
    tree = ast.parse(src, filename=path)

    assert not _imports_hidden_state(tree), f"{path} imports HiddenState"

    for node in ast.walk(tree):
        # rec.hidden / record.hidden / anything.hidden
        if isinstance(node, ast.Attribute):
            if node.attr == "hidden" or node.attr in HIDDEN_FIELDS:
                raise AssertionError(
                    f"{path}:{node.lineno} reads ground truth via .{node.attr}"
                )
        # rec["hidden"] / d["true_reason"]
        if isinstance(node, ast.Subscript):
            key = getattr(node.slice, "value", None)
            if isinstance(key, str) and (key == "hidden" or key in HIDDEN_FIELDS):
                raise AssertionError(
                    f"{path}:{node.lineno} reads ground truth via [{key!r}]"
                )
        # getattr(rec, "hidden")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "getattr" and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and (arg.value == "hidden" or arg.value in HIDDEN_FIELDS):
                raise AssertionError(
                    f"{path}:{node.lineno} reads ground truth via getattr"
                )
