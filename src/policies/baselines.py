"""
B0 and B1: the floors.

B0 does nothing, establishing what passivity is worth (exactly zero by
construction, since ABANDON has no cost and no reward).

B1 is the naive retry every subscription business ships first: a fixed number
of attempts at a fixed interval, blind to why the payment failed. It exists to
show what reason-awareness is actually worth in B2.
"""

from __future__ import annotations

from ..schema import ABANDON, RETRY_LATER


class B0DoNothing:
    NAME = "B0 do-nothing"

    def propose(self, obs):
        return [(ABANDON, 0)]


class B1NaiveRetry:
    NAME = "B1 naive retry"

    def __init__(self, attempts: int = 3, interval_hours: int = 24):
        self.attempts = attempts
        self.interval_hours = interval_hours

    def propose(self, obs):
        if obs.attempts_used >= self.attempts:
            return [(ABANDON, 0)]
        return [(RETRY_LATER, self.interval_hours)]
