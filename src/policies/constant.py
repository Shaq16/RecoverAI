"""
Feature-blind policies used only by the degeneracy check.

These ignore every input. If one of them captures a large share of oracle
value, the benchmark is measuring luck rather than judgement, and no B2/B3
result from it means anything.
"""

from __future__ import annotations

from ..schema import ABANDON, RETRY, RETRY_LATER, REQUEST_PAYMENT_UPDATE


class _Const:
    def __init__(self, name, fn):
        self.NAME = name
        self._fn = fn

    def propose(self, obs):
        return [self._fn(obs)]


CONSTANT_POLICIES = [
    _Const("always ABANDON",       lambda o: (ABANDON, 0)),
    _Const("always UPDATE now",    lambda o: (REQUEST_PAYMENT_UPDATE, 0)),
    _Const("always RETRY now",     lambda o: (RETRY, 0)),
    _Const("always RETRY @24h",    lambda o: (RETRY_LATER, 24)),
    _Const("always RETRY @48h",    lambda o: (RETRY_LATER, 48)),
    _Const("always RETRY @120h",   lambda o: (RETRY_LATER, 120)),
    _Const("UPDATE then RETRY@48", lambda o: (REQUEST_PAYMENT_UPDATE, 0)
           if o.update_requests_made == 0 else (RETRY_LATER, 48)),
]
