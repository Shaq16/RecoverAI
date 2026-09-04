# RecoverAI

A rules-first, AI-assisted recovery system for failed recurring payments. When a
subscription charge fails, RecoverAI decides whether to retry, retry later, ask
the customer to fix their payment method, or stop — subject to compliance,
retry limits and unit economics.

**Thesis:** AI should not make every recovery decision. It should earn its cost
only where deterministic rules run out of certainty.

Razorpay AI Buildathon — Track 03, AI Revenue Recovery.

---

## Why this repo leads with the benchmark, not the agent

Any competent engineer with an LLM can produce a recovery agent that runs. The
hard part is proving it does anything useful. So the first thing built here was
the thing that could prove the agent *worthless*:

1. **A frozen simulator** (`src/simulator.py`) that defines whether a given
   action would have succeeded, written and committed before any policy existed.
   Tuning a world model after seeing policy scores is training on the test set.

2. **An oracle** (`src/oracle.py`) that derives the optimal action by searching
   the action space with full knowledge of the hidden state. The obvious
   alternative — hand-labelling `optimal_action` per record — is circular,
   because the same intuitions end up in the rule baseline, which then scores
   near-perfectly by construction.

3. **A degeneracy check** (`scripts/validate_benchmark.py`) that runs policies
   which ignore every input feature. If a constant policy scores well, the
   benchmark is broken.

Current result of (3) on 1,200 records:

| constant policy | % of oracle value captured |
|---|---|
| always ABANDON | 0.0% |
| always UPDATE now | 33.1% |
| always RETRY @48h | 39.3% |
| UPDATE then RETRY @48h | 40.0% |

The best mindless policy captures 40%. The remaining 60% is what a real policy
has to earn.

## The policy ladder

| policy | description |
|---|---|
| B0 | Do nothing. Establishes the passive floor. |
| B1 | Naive retry: 3 attempts, 24h apart, reason-blind. |
| B2 | Reason-aware deterministic rules. No LLM. |
| B3 | RecoverAI. Rules for clear cases, LLM for the ambiguous tail. Routes on B2's top-two action-class EV margin; every recommendation passes compliance, economic and retry-budget gates before execution. |
| B★ | Oracle. Upper bound with full hidden knowledge. |

Scores are reported as **share of oracle-achievable value**, not raw recovery
rate. "B2 captures 71%, B3 captures 78%" is a more honest statement than two
percentages with no ceiling attached.

B2 is deliberately built to be strong. If B3 does not beat it, that is a
finding worth reporting, not a failure to hide.

## Architecture

```
failed payment
      │
      ▼
 deterministic router ──── clear ────▶ rule decision
      │
   ambiguous
      │
      ▼
  LLM decision  ──▶ compliance gate ──▶ economic gate ──▶ retry-budget gate
                          │                                      │
                       denied ◀──────────────────────────────────┘
                          │
                          ▼
                    final action + audit record
```

The LLM recommends. The policy engine decides whether the recommendation is
allowed. **The LLM never directly executes a money action**, and the gates are
enforced in code, not in a prompt.

## Benchmark design

1,200 synthetic failed recurring payments. Policies see only `Observation`;
`HiddenState` is used exclusively by the simulator and oracle. Observed decline
codes are deliberately lossy — several true causes collapse into
`DO_NOT_HONOUR`, exactly as in production.

### Adversarial slices

| slice | what it tests |
|---|---|
| `looks_alive_is_dead` | Long-tenured customer, spotless history, but the mandate was silently revoked and the merchant's cached state is stale. Punishes over-weighting customer quality. |
| `compliance_trap` | The commercially obvious action is prohibited (missing pre-debit notice, retry cap reached, above AFA threshold, outside messaging hours, active dispute). Correct behaviour is *not* recovering the money. |
| `cost_trap` | Low-value plan needing an expensive repair from a disengaged customer. Correct action is usually to stop. |
| `ambiguous_dnh` | `DO_NOT_HONOUR` with the true cause split across funds, risk and issuer downtime. No policy can be right every time; tests calibrated uncertainty. |
| `ordinary` | The other 55%. |

Metrics are reported per slice as well as overall, with bootstrap confidence
intervals, because a 5-point difference on a 60-record slice is noise.

## Results

Test split, n=564, seed 7. Exact evaluation -- no sampling, so these numbers
reproduce byte-for-byte. Full output in `results/ladder.md`.

| policy | % of oracle | 95% CI | oracle agreement | regret (INR) | waste (INR) |
|---|---|---|---|---|---|
| B0 do-nothing | 0.0% | [0.0, 0.0] | 15% | 123,584 | 0 |
| B1 naive retry | 33.4% | [26.2, 41.0] | 35% | 82,368 | 160 |
| B2 rules | **98.2%** | [97.5, 98.7] | 72% | 2,273 | 136 |
| B3 router | 96.6% | [94.6, 98.0] | 74% | 4,167 | 136 |
| B* oracle | 100% | -- | 100% | 0 | -- |

### B3 does not beat B2, and that is the result

Routing the ambiguous tail to an LLM **lost INR 2,067 net** against B2. We are
reporting it rather than tuning until it inverts.

The loss decomposes cleanly, and not in the direction you would guess:

| what the LLM changed | n | value delta | oracle agreement B2 -> B3 |
|---|---|---|---|
| same action class, different timing | 72 | **-1,477** | 64% -> 64% |
| different action class | 60 | -373 | **37% -> 55%** |
| identical to B2 | 27 | -46 | 85% -> 85% |
| rejected by the gates, fell back to B2 | 83 | 0 | 49% -> 49% |

When the model changed *which lever to pull* it was right substantially more
often -- oracle agreement on those records rose 18 points -- and still lost
money. The damage came from *timing*: it proposes sensible round numbers, while
B2 grid-searches the delay against its salary-cycle belief, and on this
benchmark the delay is worth more than the verb.

The negative result is not an artefact of the routing threshold. B3 fails to
break even at every threshold tested, and gets monotonically worse as more
traffic is routed:

| relative-margin threshold | routed | B3 % of oracle | net vs B2 (INR) |
|---|---|---|---|
| 0.02 | 33.3% | 97.3% | -1,159 |
| 0.05 | 42.4% | 97.2% | -1,264 |
| 0.10 | 44.5% | 96.7% | -2,007 |
| **0.15 (pre-registered)** | 46.5% | 96.6% | **-2,067** |
| 0.30 | 51.6% | 96.2% | -2,569 |
| 0.60 | 55.5% | 96.2% | -2,697 |

Two caveats stated plainly. These runs use `MockLLMClient`, a deterministic
offline stand-in; the real-model number is not yet measured, and
`AnthropicLLMClient` is written but unexercised. And share-of-oracle is a weak
discriminator here -- degrading B2's belief model badly moves it 98.2% -> 98.1%
-- which is why oracle agreement, regret and waste are reported alongside it.

### What the benchmark says about the thesis

The thesis was that AI should earn its cost only where deterministic rules run
out of certainty. On this benchmark the honest reading is stronger than
planned: **the rules do not run out of certainty often enough to pay for the
model.** A well-built rules engine leaves INR 2,273 of regret across 564
records; an LLM would have to capture 7.6% of that just to cover its own
invocation cost, and instead it destroys value. The router is still the right
architecture -- it is what makes the cost visible and boundable -- but on this
world model the correct configuration routes nothing.

## Compliance

`config/compliance.yaml` holds every constraint as a deterministic constant with
a `verified:` flag and a source URL.

> **Constants marked `verified: false` are structurally realistic placeholders
> and must be checked against RBI / TRAI / NPCI primary sources before this is
> shown to anyone.** They were not sourced from a language model's recall, and
> should not be promoted to `verified: true` on that basis.

`compliance.unverified_constants()` prints everything still outstanding on every
data generation run, so the gap is impossible to forget.

## Status

- [x] Schema with enforced observable / hidden separation
- [x] Frozen simulator
- [x] Compliance gate
- [x] Oracle (expectimax over the frozen model, obeys the same gates)
- [x] Dataset generator with 4 adversarial slices
- [x] Degeneracy validation
- [x] Evaluation harness and metrics (exact, not sampled)
- [x] B0 / B1 / B2
- [x] B3 + audit trail
- [ ] Charts, video
- [ ] Verify the compliance constants marked `verified: false`

## Running

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

pytest tests/                                 # observable/hidden separation must hold
python -m src.generate --n 1200 --seed 7      # writes data/{train,test}.jsonl
python -m scripts.validate_benchmark          # must pass before trusting results
python -m scripts.run_all --split test        # the ladder, B0..B3
python -m scripts.run_all --llm anthropic     # same, against a real model
```

### Layout

```
src/            schema, frozen simulator, compliance gate, oracle, generator
src/policies/   B0-B3, LLM clients. May read Observation only; enforced by tests.
config/         compliance.yaml, economics.yaml -- every constant in one readable place
scripts/        validate_benchmark and (later) run_all
tests/          test_no_leakage.py: AST-level guard against ground-truth access
data/           generated, gitignored, reproducible from seed
results/        metrics, charts, SIMULATOR_CHANGELOG.md
```

## Known limitations

- The simulator is a model, not reality. Every number here is a claim about
  behaviour *under this model*, and the model's assumptions are in one readable
  file rather than scattered through the code.
- The oracle is optimal with respect to the simulator, so it inherits any bias
  the simulator has. It is an upper bound on achievable value, not on truth.
- Synthetic data cannot establish that these interventions work on real
  customers. It can establish that a policy reasons correctly given a stated
  world model, which is a narrower and honest claim.
