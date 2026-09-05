"""
Bounded concurrency for real-model runs, without touching frozen code.

THE CONSTRAINT
--------------
`src/evaluate.py` is frozen. It walks records sequentially and calls
`policy.propose(obs)` inline, so there is nowhere inside it to parallelise --
and rewriting it would break the reproducibility claim the whole benchmark
rests on.

THE APPROACH
------------
Concurrency is applied *before* evaluation, not during it. The frozen
`LLMClient.decide` already caches decisions by payload fingerprint, so if the
cache is populated in advance every `decide()` during the real run is a cache
hit and the sequential walk costs nothing.

Which payloads will be needed is not fully knowable up front: a decision at
step 2 depends on what the model answered at step 1. So the cache is filled in
layers:

    1. Run evaluation with a DISCOVERY client that answers from the real
       client's cache when it can, and otherwise records the payload and
       returns a placeholder without calling any API.
    2. Fetch every recorded payload concurrently and store it in the real
       client's cache.
    3. Repeat. Each pass follows a trajectory shaped by the decisions already
       cached, which reveals the next layer of payloads.
    4. Stop when a pass records no new payloads.

This terminates: every pass either adds to the cache or ends the loop, the
payload space along a bounded episode is finite, and `max_passes` is a hard
backstop.

WHY THE RESULT IS STILL DETERMINISTIC
-------------------------------------
Prewarming only populates a cache. The run that produces the reported numbers
is the ordinary sequential `evaluate.evaluate(...)` call in run_all, executed
after this returns, with every decision already resolved. Completion order of
the worker threads cannot affect it, because a decision is keyed by payload
fingerprint, not by arrival order. Audit rows are still emitted in the
evaluator's own record order.

WHAT THIS DOES NOT DO
---------------------
It does not change policy semantics, gates, economics, the prompt, the schema,
validation, or what counts as a decision node. A payload fetched here is the
same single decision the sequential run would have made; caching it means it
happens once, not twice. `client.calls` is incremented once per unique payload,
exactly as a sequential run would.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence

from ..schema import ABANDON
from .llm_client import LLMClient, LLMDecision, payload_fingerprint

DEFAULT_CONCURRENCY = 4      # conservative: providers rate-limit
MAX_PASSES = 10


class _DiscoveryClient(LLMClient):
    """
    Answers from the real client's cache, records anything it cannot answer.

    Overrides `decide` rather than `_decide` so nothing it invents is ever
    written into the real client's cache. The placeholder is deliberately
    marked malformed: the schema gate then rejects it and the router falls back
    to B2, so discovery follows a legal trajectory without pretending to be a
    model decision.
    """

    name = "discovery"

    def __init__(self, real: LLMClient):
        super().__init__()
        self.real = real
        self.misses: List[Dict] = []
        self._seen = set()

    def decide(self, payload: Dict) -> LLMDecision:
        key = payload_fingerprint(payload)
        cached = self.real._cache.get(key)
        if cached is not None:
            return cached
        if key not in self._seen:
            self._seen.add(key)
            self.misses.append(payload)
        return LLMDecision(ABANDON, 0, 0.0,
                           "discovery placeholder - not a model decision",
                           "discovery", malformed=True)

    def _decide(self, payload: Dict) -> LLMDecision:   # pragma: no cover
        raise AssertionError("discovery client must never call a provider")


def prewarm(client: LLMClient,
            discover: Callable[[LLMClient], None],
            concurrency: int = DEFAULT_CONCURRENCY,
            max_passes: int = MAX_PASSES,
            log: Optional[Callable[[str], None]] = None) -> dict:
    """
    Fill `client`'s decision cache concurrently.

    `discover` is called with a client and must run one full evaluation pass
    using it -- run_all supplies a closure that builds a fresh router and calls
    the frozen evaluator. Nothing about that evaluation is parallelised; only
    the provider calls between passes are.

    Returns a summary dict for the run report.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    passes = 0
    fetched = 0
    for passes in range(1, max_passes + 1):
        probe = _DiscoveryClient(client)
        discover(probe)
        todo = probe.misses
        if not todo:
            passes -= 1          # this pass discovered nothing
            break
        if log:
            log(f"  prewarm pass {passes}: {len(todo)} decisions "
                f"({concurrency} at a time)")

        results: List[LLMDecision] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(client._decide, p): p for p in todo}
            for fut in as_completed(futures):
                payload = futures[fut]
                try:
                    decision = fut.result()
                except BaseException as e:      # noqa: BLE001
                    # A provider exception that escaped the adapter is still a
                    # failure of this run, not something to swallow.
                    decision = LLMDecision(ABANDON, 0, 0.0,
                                           f"{type(e).__name__}: {e}"[:300],
                                           getattr(client, "model", "?"),
                                           malformed=True)
                # Written under the same key the sequential path would use, so
                # the later evaluation is a pure cache hit.
                client._cache[payload_fingerprint(payload)] = decision
                client.calls += 1
                fetched += 1
                results.append(decision)

        # `.errors` is incremented inside worker threads, where `+= 1` on an
        # int is not atomic and can undercount. Undercounting would weaken the
        # fail-loud guard, so the count is reconciled here against the
        # decisions themselves, in this thread.
        malformed = sum(1 for d in results if d.malformed)
        if getattr(client, "errors", 0) < malformed:
            client.errors = malformed

    return {
        "passes": passes,
        "decisions_fetched": fetched,
        "concurrency": concurrency,
        "retries": getattr(client, "retries", 0),
        "errors": getattr(client, "errors", 0),
    }
