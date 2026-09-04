# Results

split=`test`, n=564, seed=7. Exact evaluation, no sampling.

| policy | net value (INR) | % of oracle | 95% CI | oracle agreement | regret (INR) | waste (INR) | illegal/record |
|---|---|---|---|---|---|---|---|
| B0 do-nothing | 0 | 0.0% | [0.0, 0.0] | 15% | 123,584 | 0 | 0.00 |
| B1 naive retry | 41,216 | 33.4% | [26.2, 41.0] | 35% | 82,368 | 160 | 0.45 |
| B2 rules | 121,311 | 98.2% | [97.5, 98.7] | 72% | 2,273 | 136 | 0.00 |
| B3 router | 119,416 | 96.6% | [94.6, 98.0] | 74% | 4,167 | 136 | 0.00 |

## Per slice (% of oracle value)

| slice | n | oracle value | B0 do-nothing | B1 naive retry | B2 rules | B3 router |
|---|---|---|---|---|---|---|
| `ordinary` | 316 | 76,229 | 0.0% | 40.0% | 98.4% | 95.9% |
| `looks_alive_is_dead` | 61 | 4,644 | 0.0% | -6.7% | 85.1% | 87.2% |
| `compliance_trap` | 56 | 30,235 | 0.0% | 5.9% | 99.9% | 99.7% |
| `cost_trap` | 62 | 34 | 0.0% | -280.9% | 28.1% | 28.1% |
| `ambiguous_dnh` | 69 | 12,442 | 0.0% | 75.3% | 97.8% | 97.5% |

## B3 economics

LLM backend: `mock`. Routing thresholds are pre-registered (relative margin < 0.15, stake >= 1.05 INR) and were not tuned against results.

| quantity | value |
|---|---|
| records routed to the LLM | 262 / 564 (46.5%) |
| decision nodes routed | 709 (expected 492.9 after reach weighting) |
| recommendations passing all gates | 417 / 709 (59%) |
| B2 regret | INR 2,273 |
| B3 regret | INR 4,167 |
| regret captured vs B2 | INR -1,894 (-83.3%) |
| LLM cost @ INR 0.35/decision | INR 173 |
| capture needed to break even | 7.6% of B2 regret |
| **net benefit vs B2** | **INR -2,067** |
