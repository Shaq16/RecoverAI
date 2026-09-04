"""
Run the policy ladder and write results.

    python -m scripts.run_all [--split test] [--n 1200] [--seed 7]

Everything here is exact and reproducible: the same seed gives the same table,
because policy evaluation does not sample. The only stochastic element is the
bootstrap over records, which is itself seeded.

Writes results/metrics.json and results/ladder.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src import compliance, evaluate, generate, oracle
from src.policies.baselines import B0DoNothing, B1NaiveRetry
from src.policies.b2_rules import B2Rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def build_ladder(econ, cfg):
    return [B0DoNothing(), B1NaiveRetry(), B2Rules(econ, cfg)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = generate.build(args.n, args.seed)
    if args.split != "all":
        recs = [r for r in recs if r.split == args.split]

    print(f"split={args.split}  n={len(recs)}  "
          f"oracle achievable: INR {sum(r.hidden.oracle_ev for r in recs):,.0f}")
    print(f"simulator: {__import__('src.simulator', fromlist=['x']).SIMULATOR_VERSION}\n")

    metrics = []
    for pol in build_ladder(econ, cfg):
        m = evaluate.evaluate(recs, pol, econ=econ, cfg=cfg, bootstrap=args.bootstrap)
        metrics.append(m)
        print(f"  scored {m.name}")

    print("\n" + "=" * 78)
    print("POLICY LADDER -- share of oracle-achievable value")
    print("=" * 78)
    print(evaluate.format_table(metrics))
    print("\nPER SLICE")
    print("-" * 78)
    print(evaluate.format_slice_table(metrics))

    unver = compliance.unverified_constants(cfg)
    if unver:
        print(f"\n[!] {len(unver)} compliance constants still unverified "
              f"-- results are conditional on placeholder values.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({
            "split": args.split, "n": len(recs), "seed": args.seed,
            "unverified_constants": unver,
            "policies": [m.to_dict() for m in metrics],
        }, f, indent=2)

    with open(os.path.join(RESULTS, "ladder.md"), "w") as f:
        f.write(f"# Results\n\nsplit=`{args.split}`, n={len(recs)}, seed={args.seed}. "
                f"Exact evaluation, no sampling.\n\n")
        f.write("| policy | net value (INR) | % of oracle | 95% CI | oracle agreement | regret (INR) | waste (INR) | illegal/record |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for m in metrics:
            f.write(f"| {m.name} | {m.policy_value:,.0f} | {100*m.share:.1f}% | "
                    f"[{100*m.ci_low:.1f}, {100*m.ci_high:.1f}] | "
                    f"{100*m.root_agreement:.0f}% | {m.regret:,.0f} | "
                    f"{m.waste:,.0f} | {m.denied_rate:.2f} |\n")
        f.write("\n## Per slice (% of oracle value)\n\n| slice | n | oracle value |")
        for m in metrics:
            f.write(f" {m.name} |")
        f.write("\n|---|---|---|" + "---|" * len(metrics) + "\n")
        for t, ref in metrics[0].per_slice.items():
            f.write(f"| `{t}` | {ref.n} | {ref.oracle_value:,.0f} |")
            for m in metrics:
                s = m.per_slice.get(t)
                f.write(f" {100*s.share:.1f}% |" if s else " - |")
            f.write("\n")
    print(f"\nwrote {RESULTS}/metrics.json and {RESULTS}/ladder.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
