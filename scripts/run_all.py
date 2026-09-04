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
from src.policies.b3_router import B3Router
from src.policies.llm_client import AnthropicLLMClient, MockLLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def build_ladder(econ, cfg, router):
    return [B0DoNothing(), B1NaiveRetry(), B2Rules(econ, cfg), router]


def llm_accounting(b3: evaluate.PolicyMetrics, router: B3Router,
                   b2: evaluate.PolicyMetrics, econ: dict) -> dict:
    """
    What the LLM actually cost, and whether it paid for itself.

    A decision at step k only happens if every earlier attempt in that episode
    failed, so invocations are weighted by reach probability -- the product of
    (1 - p_success) over prior steps, read off the audit trail. Counting every
    node at full price would overstate the bill for exactly the deep retries
    that usually never happen.
    """
    unit = econ["costs"]["llm_decision"]
    nodes = accepted = routed_records = 0
    expected = 0.0

    for r in b3.results:
        reach, routed = 1.0, False
        for step in r.audit:
            d = router.decisions.get((r.payment_id, step.elapsed_hours))
            if d is not None and d.llm_invoked:
                nodes += 1
                expected += reach
                routed = True
                if d.llm_accepted:
                    accepted += 1
            reach *= (1.0 - step.p_success)
        if routed:
            routed_records += 1

    cost = expected * unit
    gain = b3.policy_value - b2.policy_value
    return {
        "routed_records": routed_records,
        "invocation_rate": routed_records / b3.n if b3.n else 0.0,
        "decision_nodes_routed": nodes,
        "expected_invocations": expected,
        "accepted_by_gates": accepted,
        "acceptance_rate": accepted / nodes if nodes else 0.0,
        "unique_client_calls": router.client.calls,
        "llm_unit_cost": unit,
        "llm_cost": cost,
        "value_vs_b2": gain,
        "net_benefit_vs_b2": gain - cost,
        "regret_b2": b2.regret,
        "regret_b3": b3.regret,
        "regret_captured": b2.regret - b3.regret,
        "regret_captured_share": ((b2.regret - b3.regret) / b2.regret
                                  if abs(b2.regret) > 1e-9 else 0.0),
        "breakeven_capture_needed": cost / b2.regret if abs(b2.regret) > 1e-9 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--llm", default="mock", choices=["mock", "anthropic"],
                    help="mock needs no API key and is deterministic")
    ap.add_argument("--model", default="claude-opus-5")
    args = ap.parse_args()

    econ, cfg = oracle.load_economics(), compliance.load_config()
    recs = generate.build(args.n, args.seed)
    if args.split != "all":
        recs = [r for r in recs if r.split == args.split]

    print(f"split={args.split}  n={len(recs)}  "
          f"oracle achievable: INR {sum(r.hidden.oracle_ev for r in recs):,.0f}")
    print(f"simulator: {__import__('src.simulator', fromlist=['x']).SIMULATOR_VERSION}\n")

    client = (AnthropicLLMClient(model=args.model) if args.llm == "anthropic"
              else MockLLMClient())
    router = B3Router(econ, cfg, client=client)
    print(f"B3 LLM backend: {client.name}"
          f"{' (' + args.model + ')' if args.llm == 'anthropic' else ''}\n")

    metrics = []
    for pol in build_ladder(econ, cfg, router):
        # B3 needs the audit trail to weight invocations by reach probability.
        m = evaluate.evaluate(recs, pol, econ=econ, cfg=cfg,
                              bootstrap=args.bootstrap,
                              collect_audit=isinstance(pol, B3Router))
        metrics.append(m)
        print(f"  scored {m.name}")

    by_name = {m.name: m for m in metrics}
    acct = llm_accounting(by_name["B3 router"], router, by_name["B2 rules"], econ)

    print("\n" + "=" * 78)
    print("POLICY LADDER -- share of oracle-achievable value")
    print("=" * 78)
    print(evaluate.format_table(metrics))
    print("\nPER SLICE")
    print("-" * 78)
    print(evaluate.format_slice_table(metrics))

    print("\n" + "=" * 78)
    print("B3 ECONOMICS -- did routing to a model pay for itself?")
    print("=" * 78)
    print(f"  records routed to the LLM      {acct['routed_records']} / {len(recs)}"
          f"  ({100*acct['invocation_rate']:.1f}%)")
    print(f"  decision nodes routed          {acct['decision_nodes_routed']}"
          f"  (expected {acct['expected_invocations']:.1f} after reach weighting)")
    print(f"  recommendations passing gates  {acct['accepted_by_gates']}"
          f" / {acct['decision_nodes_routed']}"
          f"  ({100*acct['acceptance_rate']:.0f}%)")
    print(f"  unique client calls (cached)   {acct['unique_client_calls']}")
    print()
    print(f"  B2 regret                      INR {acct['regret_b2']:>10,.0f}")
    print(f"  B3 regret                      INR {acct['regret_b3']:>10,.0f}")
    print(f"  regret captured vs B2          INR {acct['regret_captured']:>10,.0f}"
          f"  ({100*acct['regret_captured_share']:+.1f}% of B2 regret)")
    print(f"  LLM cost @ INR {acct['llm_unit_cost']}/decision      "
          f"INR {acct['llm_cost']:>10,.0f}")
    print(f"  capture needed to break even   "
          f"{100*acct['breakeven_capture_needed']:.1f}% of B2 regret")
    print("  " + "-" * 52)
    print(f"  NET BENEFIT vs B2              INR {acct['net_benefit_vs_b2']:>10,.0f}"
          f"   {'PAYS FOR ITSELF' if acct['net_benefit_vs_b2'] > 0 else 'DOES NOT PAY'}")

    unver = compliance.unverified_constants(cfg)
    if unver:
        print(f"\n[!] {len(unver)} compliance constants still unverified "
              f"-- results are conditional on placeholder values.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({
            "split": args.split, "n": len(recs), "seed": args.seed,
            "unverified_constants": unver,
            "llm_backend": client.name,
            "b3_economics": acct,
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
        f.write("\n## B3 economics\n\n")
        f.write(f"LLM backend: `{client.name}`. Routing thresholds are "
                f"pre-registered (relative margin < {router.ambiguity_threshold}, "
                f"stake >= {router.min_stake:.2f} INR) and were not tuned "
                f"against results.\n\n")
        f.write("| quantity | value |\n|---|---|\n")
        f.write(f"| records routed to the LLM | {acct['routed_records']} / "
                f"{len(recs)} ({100*acct['invocation_rate']:.1f}%) |\n")
        f.write(f"| decision nodes routed | {acct['decision_nodes_routed']} "
                f"(expected {acct['expected_invocations']:.1f} after reach "
                f"weighting) |\n")
        f.write(f"| recommendations passing all gates | "
                f"{acct['accepted_by_gates']} / {acct['decision_nodes_routed']} "
                f"({100*acct['acceptance_rate']:.0f}%) |\n")
        f.write(f"| B2 regret | INR {acct['regret_b2']:,.0f} |\n")
        f.write(f"| B3 regret | INR {acct['regret_b3']:,.0f} |\n")
        f.write(f"| regret captured vs B2 | INR {acct['regret_captured']:,.0f} "
                f"({100*acct['regret_captured_share']:+.1f}%) |\n")
        f.write(f"| LLM cost @ INR {acct['llm_unit_cost']}/decision | "
                f"INR {acct['llm_cost']:,.0f} |\n")
        f.write(f"| capture needed to break even | "
                f"{100*acct['breakeven_capture_needed']:.1f}% of B2 regret |\n")
        f.write(f"| **net benefit vs B2** | **INR "
                f"{acct['net_benefit_vs_b2']:,.0f}** |\n")

    print(f"\nwrote {RESULTS}/metrics.json and {RESULTS}/ladder.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
