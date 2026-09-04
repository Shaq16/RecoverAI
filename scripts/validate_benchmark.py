"""
Benchmark sanity check. Run this before trusting any policy result.

A benchmark is only meaningful if a mindless policy cannot score well on it.
This script runs several constant policies -- ones that ignore every input
feature -- and reports what fraction of the oracle's achievable value they
capture. If any constant policy captures a large share, the benchmark is too
easy and the economics or slice mix need rebalancing before B2/B3 results mean
anything.

    python -m scripts.validate_benchmark

WHY THIS FILE WAS REWRITTEN
---------------------------
The original version sampled outcomes: three trials per record, drawn from a
single `random.Random(99)` shared across every record and every policy. That
was wrong in two ways, and the errors were not small.

  * The headline was sampling error. It reported the best constant policy at
    40.0% of oracle value. Exact evaluation puts that same policy at 36.9% --
    the published figure was overstated by 3.1 points. Re-seeding alone moved
    the number between 31.1% and 40.0%; the committed 40.0% was the luckiest
    cell in that range.

  * The ranking was wrong. It named "UPDATE then RETRY@48" the strongest
    mindless policy. It is not: exact evaluation ranks it third, behind both
    "always RETRY @48h" (39.8%, the true best) and "always RETRY @24h" (37.6%).
    A degeneracy check that misidentifies the strongest degenerate policy is
    not doing its job.

Both faults had the same root cause. Each policy reset the RNG to the same seed
but consumed a different number of draws per record, so the streams
desynchronised across policies; and three trials is far too few to average
anything. At fifteen trials the same policy converged to ~36.5%.

There is no need to sample at all. `simulator.success_probability` is
deterministic and the episode tree is bounded, so `src.evaluate` computes each
policy's expected value in closed form by recursing both outcome branches. This
script now uses that same evaluator -- the one the main benchmark already
trusts -- and the constant policies already defined in `src.policies.constant`,
rather than maintaining a second inline set of lambdas.

**The exact figures below are the authoritative ones. The previous 40.0% was
sampling error and should not be cited.**
"""

from __future__ import annotations

import argparse
import sys

from src import compliance, evaluate, generate, oracle
from src.policies.constant import CONSTANT_POLICIES

WARN_THRESHOLD = 0.60  # a constant policy above this = benchmark too easy

# What the superseded Monte-Carlo implementation reported, for the record.
# Kept so the correction is visible rather than quietly rewritten. Policies it
# never exercised are marked None.
SUPERSEDED_MONTE_CARLO = {
    "always ABANDON": 0.0,
    "always UPDATE now": 0.331,
    "always RETRY now": None,
    "always RETRY @24h": 0.365,
    "always RETRY @48h": 0.393,
    "always RETRY @120h": 0.361,
    "UPDATE then RETRY@48": 0.400,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = generate.build(args.n, args.seed)
    oracle_total = sum(r.hidden.oracle_ev for r in recs)

    print(f"records: {len(recs)}   "
          f"oracle achievable net value: INR {oracle_total:,.0f}")
    print("evaluation: exact (both outcome branches recursed; no sampling)\n")

    print(f"{'constant policy':<24}{'net value':>14}{'% of oracle':>14}"
          f"{'was (MC)':>11}{'delta':>9}")
    print("-" * 72)

    worst = 0.0
    for pol in CONSTANT_POLICIES:
        m = evaluate.evaluate(recs, pol, econ=econ, cfg=cfg, bootstrap=0)
        worst = max(worst, m.share)
        old = SUPERSEDED_MONTE_CARLO.get(pol.NAME)
        if old is None:
            was, delta = "not run", ""
        else:
            was = f"{100 * old:.1f}%"
            delta = f"{100 * (m.share - old):+.1f}"
        print(f"{pol.NAME:<24}{m.policy_value:>14,.0f}"
              f"{100 * m.share:>13.1f}%{was:>11}{delta:>9}")

    print("-" * 72)
    print(f"best constant policy captures {100 * worst:.1f}% of oracle value")
    print("  (the superseded Monte-Carlo run reported 40.0% and named the wrong "
          "policy;\n   that was sampling error -- the figure above is "
          "authoritative)")

    if worst > WARN_THRESHOLD:
        print("\n[FAIL] Benchmark is too easy. Rebalance before trusting B2/B3.")
        return 1
    print("[OK] Headroom for a real policy to earn its keep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
