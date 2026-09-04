"""
Dataset generator.

Produces synthetic failed recurring payments with a hidden ground truth that
policies never see. Adversarial slices are constructed deliberately so the
benchmark can embarrass the system rather than flatter it.

The optimal action is NOT hand-authored here. It is computed by the oracle
(src/oracle.py) searching the frozen simulator. See oracle.py for why.

Usage:
    python -m src.generate --n 1200 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

from . import compliance, oracle
from .schema import (
    Observation, HiddenState, Record,
    INSUFFICIENT_FUNDS, ISSUER_DOWNTIME, GATEWAY_TIMEOUT, RISK_DECLINE,
    MANDATE_REVOKED, MANDATE_EXPIRED, CARD_EXPIRED, AMOUNT_EXCEEDS_CAP,
    DO_NOT_HONOUR,
    SLICE_ORDINARY, SLICE_LOOKS_ALIVE_IS_DEAD, SLICE_COMPLIANCE_TRAP,
    SLICE_COST_TRAP, SLICE_AMBIGUOUS_DNH,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

SLICE_MIX = {
    SLICE_ORDINARY: 0.55,
    SLICE_LOOKS_ALIVE_IS_DEAD: 0.11,
    SLICE_COMPLIANCE_TRAP: 0.11,
    SLICE_COST_TRAP: 0.11,
    SLICE_AMBIGUOUS_DNH: 0.12,
}

ORDINARY_REASONS = [
    (INSUFFICIENT_FUNDS, 0.34), (ISSUER_DOWNTIME, 0.12), (GATEWAY_TIMEOUT, 0.10),
    (RISK_DECLINE, 0.12), (CARD_EXPIRED, 0.12), (MANDATE_EXPIRED, 0.09),
    (MANDATE_REVOKED, 0.06), (AMOUNT_EXCEEDS_CAP, 0.05),
]

# How a true reason presents itself to the merchant. Production decline codes
# are lossy; DO_NOT_HONOUR is the catch-all that hides several real causes.
OBSERVED_MAP = {
    INSUFFICIENT_FUNDS: [(INSUFFICIENT_FUNDS, 0.70), (DO_NOT_HONOUR, 0.30)],
    RISK_DECLINE:       [(RISK_DECLINE, 0.45), (DO_NOT_HONOUR, 0.55)],
    ISSUER_DOWNTIME:    [(ISSUER_DOWNTIME, 0.60), (DO_NOT_HONOUR, 0.40)],
    GATEWAY_TIMEOUT:    [(GATEWAY_TIMEOUT, 0.85), (DO_NOT_HONOUR, 0.15)],
    CARD_EXPIRED:       [(CARD_EXPIRED, 0.90), (DO_NOT_HONOUR, 0.10)],
    MANDATE_EXPIRED:    [(MANDATE_EXPIRED, 0.85), (DO_NOT_HONOUR, 0.15)],
    MANDATE_REVOKED:    [(MANDATE_REVOKED, 0.70), (DO_NOT_HONOUR, 0.30)],
    AMOUNT_EXCEEDS_CAP: [(AMOUNT_EXCEEDS_CAP, 0.80), (DO_NOT_HONOUR, 0.20)],
}

PLAN_VALUES = [49, 79, 99, 149, 199, 299, 399, 499, 799, 999, 1499, 2499, 4999]


def _weighted(rng: random.Random, pairs):
    r, acc = rng.random(), 0.0
    for item, w in pairs:
        acc += w
        if r <= acc:
            return item
    return pairs[-1][0]


def _hours_to_salary(day_of_month: int, rng: random.Random) -> int:
    """Money typically lands around month start. Everything else is a wait."""
    if day_of_month >= 25:
        base = (32 - day_of_month) * 24
    elif day_of_month <= 3:
        base = rng.randint(2, 30)
    else:
        base = (31 - day_of_month) * 24
    if rng.random() < 0.22:          # windfall / transfer from elsewhere
        base = rng.randint(4, 60)
    return max(1, int(base + rng.gauss(0, 8)))


def _base_customer(rng: random.Random):
    tenure = int(max(5, rng.lognormvariate(5.2, 0.85)))
    cycles = max(1, min(tenure // 30, 36))
    fail_rate = min(0.6, max(0.0, rng.gauss(0.12, 0.12)))
    prior_failures = int(cycles * fail_rate)
    prior_successes = max(0, cycles - prior_failures)
    return tenure, prior_successes, prior_failures


def _make(rng: random.Random, idx: int, slice_tag: str) -> Record:
    tenure, succ, fail = _base_customer(rng)
    day_of_month = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    plan = rng.choice(PLAN_VALUES)
    amount = float(plan)
    method = _weighted(rng, [("upi_autopay", 0.5), ("card", 0.36), ("enach", 0.14)])

    # ---------- slice-specific construction ----------
    if slice_tag == SLICE_ORDINARY:
        reason = _weighted(rng, ORDINARY_REASONS)

    elif slice_tag == SLICE_LOOKS_ALIVE_IS_DEAD:
        # Excellent customer on paper; mandate was silently revoked and the
        # merchant's cached mandate state is stale. History misleads badly.
        reason = MANDATE_REVOKED
        tenure = rng.randint(400, 1400)
        succ, fail = rng.randint(12, 40), rng.randint(0, 1)

    elif slice_tag == SLICE_COST_TRAP:
        # Low-value plan needing an expensive repair from a disengaged user.
        # Correct answer is usually to stop, even though it is "recoverable".
        reason = rng.choice([CARD_EXPIRED, MANDATE_EXPIRED, AMOUNT_EXCEEDS_CAP])
        amount = float(rng.choice([29, 39, 49, 59, 79]))
        tenure = rng.randint(20, 120)
        succ, fail = rng.randint(0, 2), rng.randint(1, 4)

    elif slice_tag == SLICE_AMBIGUOUS_DNH:
        reason = rng.choice([INSUFFICIENT_FUNDS, RISK_DECLINE, ISSUER_DOWNTIME])

    elif slice_tag == SLICE_COMPLIANCE_TRAP:
        reason = _weighted(rng, [(INSUFFICIENT_FUNDS, 0.55), (GATEWAY_TIMEOUT, 0.25),
                                 (ISSUER_DOWNTIME, 0.20)])
    else:
        raise ValueError(slice_tag)

    # ---------- observable presentation ----------
    if slice_tag == SLICE_AMBIGUOUS_DNH:
        decline_code = DO_NOT_HONOUR
    else:
        decline_code = _weighted(rng, OBSERVED_MAP[reason])

    if slice_tag == SLICE_LOOKS_ALIVE_IS_DEAD:
        mandate_status = "active"          # stale record: the trap
        decline_code = DO_NOT_HONOUR
    elif reason == MANDATE_REVOKED:
        mandate_status = "revoked"
    elif reason == MANDATE_EXPIRED:
        mandate_status = "expired"
    else:
        mandate_status = _weighted(rng, [("active", 0.93), ("unknown", 0.07)])

    if reason == AMOUNT_EXCEEDS_CAP:
        mandate_cap = round(amount * rng.uniform(0.45, 0.9), 2)
    else:
        mandate_cap = round(amount * rng.uniform(1.2, 3.0), 2)

    pre_debit_sent = rng.random() < 0.88
    attempts_already = _weighted(rng, [(0, 0.82), (1, 0.13), (2, 0.05)])
    active_dispute = rng.random() < 0.04
    recent_refund = rng.random() < 0.09

    # Compliance trap: make the commercially obvious action illegal.
    if slice_tag == SLICE_COMPLIANCE_TRAP:
        kind = rng.choice(["no_notice", "near_cap", "afa", "night", "dispute"])
        if kind == "no_notice":
            pre_debit_sent = False
        elif kind == "near_cap":
            attempts_already = 3
        elif kind == "afa":
            amount = float(rng.choice([15600, 18000, 22000, 28000, 34000]))
            mandate_cap = round(amount * rng.uniform(1.2, 2.0), 2)
        elif kind == "night":
            hour = rng.choice([22, 23, 0, 1, 2])
            pre_debit_sent = False
        elif kind == "dispute":
            active_dispute = True

    obs = Observation(
        payment_id=f"pay_{idx:05d}",
        amount=amount,
        currency="INR",
        decline_code=decline_code,
        payment_method=method,
        customer_tenure_days=tenure,
        prior_successes=succ,
        prior_failures=fail,
        days_into_billing_cycle=rng.randint(0, 3),
        day_of_month=day_of_month,
        hour_of_day=hour,
        mandate_status=mandate_status,
        mandate_cap=mandate_cap,
        recent_refund=recent_refund,
        active_dispute=active_dispute,
        pre_debit_notice_sent=pre_debit_sent,
        attempts_already_made=attempts_already,
        subscription_plan_value=amount,
    )

    # ---------- hidden state ----------
    engagement = 0.45 * min(tenure / 540.0, 1.0) + 0.55 * (succ / max(succ + fail, 1))
    responsiveness = max(0.02, min(0.98, rng.gauss(engagement, 0.18)))
    if slice_tag == SLICE_COST_TRAP:
        responsiveness = max(0.02, min(0.45, responsiveness * 0.5))

    hidden = HiddenState(
        true_reason=reason,
        funds_return_hours=(_hours_to_salary(day_of_month, rng)
                            if reason == INSUFFICIENT_FUNDS else None),
        outage_duration_hours=(rng.randint(2, 20)
                               if reason == ISSUER_DOWNTIME else None),
        update_responsiveness=round(responsiveness, 4),
    )

    return Record(obs=obs, hidden=hidden, slice_tag=slice_tag)


def build(n: int, seed: int) -> List[Record]:
    rng = random.Random(seed)
    econ, cfg = oracle.load_economics(), compliance.load_config()

    tags: List[str] = []
    for tag, frac in SLICE_MIX.items():
        tags += [tag] * int(round(n * frac))
    while len(tags) < n:
        tags.append(SLICE_ORDINARY)
    tags = tags[:n]
    rng.shuffle(tags)

    records = []
    for i, tag in enumerate(tags):
        rec = _make(rng, i, tag)
        oracle.annotate(rec.obs, rec.hidden, econ, cfg)
        rec.split = "test" if rng.random() < 0.5 else "train"
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = build(args.n, args.seed)

    for split in ("train", "test"):
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for r in records:
                if r.split == split:
                    f.write(json.dumps(r.to_dict()) + "\n")
        print(f"wrote {path}: {sum(1 for r in records if r.split == split)} records")

    unver = compliance.unverified_constants()
    if unver:
        print(f"\n[!] {len(unver)} compliance constants still unverified:")
        for u in unver:
            print(f"    - {u}")


def load(path: str) -> List[Record]:
    with open(path) as f:
        return [Record.from_dict(json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    main()
