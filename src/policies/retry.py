"""
Bounded retry for transient provider failures.

WHY THIS EXISTS
---------------
Live smoke tests showed both real backends failing for reasons that have
nothing to do with the model's judgement: Gemini returned `503 UNAVAILABLE`
("this model is currently experiencing high demand") on 2 of 3 calls, and
Nemotron timed out on 1 of 3. The fail-loud guard then correctly rejected the
run -- but it was rejecting a run because Google had a busy minute, not because
anything about the benchmark was wrong.

A 503 is not a decision. Retrying it does not weaken the guard; it makes the
guard fire on what it was designed for -- an unrecoverable failure -- instead
of on infrastructure noise.

WHAT IS AND IS NOT RETRIED
--------------------------
Retried (genuinely transient, the provider is telling us to come back):
    HTTP 429  rate limited -- "Too Many Requests", slow down
    HTTP 503  service unavailable / overloaded
    connection reset, connection refused, socket timeout, read timeout

NOT retried, ever -- these are answers, not noise, and must stay fail-loud:
    HTTP 429  QUOTA EXHAUSTED -- "RESOURCE_EXHAUSTED", "exceeded your current
              quota". Same status code as a rate limit, opposite meaning: the
              budget is gone, so no number of retries can succeed and each one
              spends more of what is already spent.
    400 bad request        the request itself is wrong
    401 / 403              bad or missing credentials
    404                    wrong model or endpoint
    malformed model output  the model answered; the answer was invalid
    schema violation        likewise
    refusal                 likewise
    missing API key         raised before any call is attempted
    unsupported parameter    a 400 in practice

500, 502 and 504 are deliberately NOT retried. They are plausibly transient,
but the brief named 429 and 503 specifically and asked for conservative
behaviour; retrying less is the safer default. Widening the set is a one-line
change to TRANSIENT_STATUS if that turns out to be needed.

WHY THIS CANNOT INFLATE THE BENCHMARK
-------------------------------------
Retry lives strictly inside a provider's `_decide`, wrapping only the transport
call. The frozen `LLMClient.decide` increments `.calls` once and caches by
payload fingerprint *before* `_decide` runs, so a retry cannot become a second
billed call, a second decision node, or a second recovery. Reported cost is
derived from the audit trail's reach-weighted node count, never from attempt
counts. A separate `.retries` counter exists purely so the report can say how
much turbulence there was.
"""

from __future__ import annotations

import random
import re
import socket
import time
from typing import Callable, Optional, Set, TypeVar

T = TypeVar("T")

TRANSIENT_STATUS: Set[int] = {429, 503}

# A 429 means two different things and only one of them is worth retrying.
#
#   "Too Many Requests"  -> slow down. Backing off is exactly right.
#   "RESOURCE_EXHAUSTED" -> the quota is gone. Retrying cannot succeed, and
#                           every attempt spends more of the quota that is
#                           already exhausted.
#
# The 100-record dry run made this concrete: 140 of 209 calls returned
# `429 RESOURCE_EXHAUSTED - you exceeded your current quota`, and the retry
# logic turned them into 311 further attempts, none of which could have worked.
# Treating quota exhaustion as permanent stops the run immediately, which is
# both cheaper and more honest -- the guard should fire on a real, terminal
# failure rather than grinding through a budget that no longer exists.
QUOTA_MARKERS = (
    "resource_exhausted",
    "exceeded your current quota",
    "quota exceeded",
    "out of quota",
    "insufficient_quota",
)

# Status codes that must never be retried even if a provider's exception type
# looks generic. Listed explicitly so the intent is greppable.
PERMANENT_STATUS: Set[int] = {400, 401, 403, 404, 405, 422}

MAX_ATTEMPTS = 3          # total attempts, not retries-after-the-first
BASE_DELAY = 0.6          # seconds
MAX_DELAY = 8.0
JITTER = 0.25             # +/- fraction of the computed delay

_STATUS_RE = re.compile(r"\b(4\d\d|5\d\d)\b")

_rng = random.Random(20260905)


def status_of(exc: BaseException) -> Optional[int]:
    """
    Best-effort HTTP status for a provider exception.

    Providers disagree on where the code lives: urllib puts it on `.code`,
    google-genai on `.code` too but as part of an ApiError, and some wrappers
    only put it in the message. Checked in that order, with a message scan as
    the last resort so a `ServerError: 503 UNAVAILABLE` is still classified.
    """
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    m = _STATUS_RE.search(str(exc))
    return int(m.group(1)) if m else None


def is_quota_exhausted(exc: BaseException) -> bool:
    """
    True when a 429 is a spent quota rather than a rate limit.

    Matched on the provider's own wording, because the status code alone
    cannot distinguish the two cases.
    """
    text = str(exc).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def is_transient(exc: BaseException) -> bool:
    """True only for failures worth trying again."""
    code = status_of(exc)
    if code is not None:
        # An explicit status wins: a permanent code is never retried, even if
        # the exception is also a subclass of something timeout-ish.
        if code in PERMANENT_STATUS:
            return False
        # A rate-limit 429 is worth backing off from; an exhausted-quota 429
        # is terminal and must fail fast.
        if code == 429 and is_quota_exhausted(exc):
            return False
        return code in TRANSIENT_STATUS

    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                        ConnectionRefusedError, BrokenPipeError)):
        return True
    if isinstance(exc, ConnectionError):
        return True
    # urllib wraps socket errors in URLError; the reason carries the truth.
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException) and reason is not exc:
        return is_transient(reason)
    if isinstance(reason, str) and "timed out" in reason.lower():
        return True
    # Plain read timeouts surface as OSError with a timeout message.
    return "timed out" in str(exc).lower()


def backoff_delay(attempt: int, base: float = BASE_DELAY,
                  max_delay: float = MAX_DELAY,
                  rng: Optional[random.Random] = None) -> float:
    """
    Exponential backoff with jitter, capped.

    attempt is 1-based: the delay returned is the wait BEFORE attempt+1.
    Jitter spreads retries so concurrent workers do not synchronise into a
    thundering herd against a provider that is already overloaded.
    """
    r = rng or _rng
    raw = min(base * (2 ** (attempt - 1)), max_delay)
    return max(0.0, raw * (1.0 + r.uniform(-JITTER, JITTER)))


def with_retry(fn: Callable[[], T],
               attempts: int = MAX_ATTEMPTS,
               on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
               sleep: Callable[[float], None] = time.sleep,
               rng: Optional[random.Random] = None) -> T:
    """
    Call `fn`, retrying only transient failures, at most `attempts` times.

    Re-raises the last exception when the budget is exhausted, so the caller's
    existing failure handling -- count the error, mark the decision malformed,
    let run_all reject the run -- is reached unchanged. A permanent failure
    re-raises immediately without consuming the budget.

    `sleep` and `rng` are injectable so tests can exercise the backoff without
    waiting and without randomness.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as e:          # noqa: BLE001 - re-raised below
            last = e
            if attempt >= attempts or not is_transient(e):
                raise
            delay = backoff_delay(attempt, rng=rng)
            if on_retry is not None:
                on_retry(attempt, e, delay)
            sleep(delay)
    raise last                               # unreachable; keeps type checkers happy
