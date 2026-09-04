"""
Benchmark sanity check. Run this before trusting any policy result.

A benchmark is only meaningful if a mindless policy cannot score well on it.
This script runs several constant policies -- ones that ignore every feature --
and reports what fraction of the oracle's achievable value they capture.

If any constant policy captures a large share of oracle value, the benchmark
is too easy and the economics or slice mix need rebalancing before B2/B3
results mean anything.

    python -m scripts.validate_benchmark
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace

from src import compliance, generate, oracle
from src.schema import (
    ABANDON, RETRY_LATER, REQUEST_PAYMENT_UPDATE,
)
from src.simulator import success_probability

WARN_THRESHOLD = 0.60  # a constant policy above this = benchmark too easy


def run_episode(rec, chooser, rng, econ, cfg, margin):
    obs, hid = replace(rec.obs), rec.hidden
    cost, recovered = 0.0, False
    for _ in range(8):
        action, delay = chooser(obs)
        if action == ABANDON:
            break
        if not compliance.check(obs, action, delay, cfg).allowed:
            break
        cost += oracle.action_cost(action, econ)
        if rng.random() < success_probability(obs, hid, action, delay):
            recovered = True
            break
        if action == REQUEST_PAYMENT_UPDATE:
            obs = replace(obs, elapsed_hours=obs.elapsed_hours + delay + 1,
                          update_requests_made=obs.update_requests_made + 1)
        else:
            obs = replace(obs, elapsed_hours=obs.elapsed_hours + delay + 1,
                          attempts_used=obs.attempts_used + 1)
    return (rec.obs.amount * margin if recovered else 0.0) - cost


CONSTANT_POLICIES = {
    "always ABANDON":       lambda o: (ABANDON, 0),
    "always UPDATE now":    lambda o: (REQUEST_PAYMENT_UPDATE, 0),
    "always RETRY @24h":    lambda o: (RETRY_LATER, 24),
    "always RETRY @48h":    lambda o: (RETRY_LATER, 48),
    "always RETRY @120h":   lambda o: (RETRY_LATER, 120),
    "UPDATE then RETRY@48": lambda o: (
        (REQUEST_PAYMENT_UPDATE, 0) if o.update_requests_made == 0 else (RETRY_LATER, 48)
    ),
}


def main(n: int = 1200, seed: int = 7, trials: int = 3):
    recs = generate.build(n, seed)
    econ, cfg = oracle.load_economics(), compliance.load_config()
    margin = econ["contribution_margin"]

    oracle_total = sum(r.hidden.oracle_ev for r in recs)
    print(f"records: {n}   oracle achievable net value: INR {oracle_total:,.0f}\n")
    print(f"{'constant policy':<24}{'net value':>14}{'% of oracle':>14}")
    print("-" * 52)

    worst = 0.0
    for name, fn in CONSTANT_POLICIES.items():
        rng = random.Random(99)
        total = sum(sum(run_episode(r, fn, rng, econ, cfg, margin)
                        for _ in range(trials)) / trials for r in recs)
        share = total / oracle_total
        worst = max(worst, share)
        print(f"{name:<24}{total:>14,.0f}{100 * share:>13.1f}%")

    print("-" * 52)
    print(f"best constant policy captures {100 * worst:.1f}% of oracle value")
    if worst > WARN_THRESHOLD:
        print("\n[FAIL] Benchmark is too easy. Rebalance before trusting B2/B3.")
        return 1
    print("[OK] Headroom for a real policy to earn its keep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
