"""
NVIDIA Nemotron 3.5 Lightning backend for B3, via OpenRouter.

    export OPENROUTER_API_KEY=...
    python -m scripts.run_all --split test --llm nemotron

WHY A SEPARATE MODULE
---------------------
`llm_client.py` is frozen: it owns MockLLMClient, the prompt, the response
schema and the coercion every backend shares, and the benchmark's
reproducibility claim rests on it not moving. A provider is added by
subclassing from the outside, exactly as `gemini_client.py` does.

WHY FORCED TOOL CALLING RATHER THAN response_format
---------------------------------------------------
This was checked against the live OpenRouter models API rather than assumed.
`nvidia/nemotron-3.5-lightning:free` reports:

    supported_parameters: include_reasoning, max_tokens, reasoning, seed,
                          temperature, tool_choice, tools, top_p

`response_format` is NOT in that list, and OpenRouter's documented behaviour
for an unsupported parameter is to fail the request. So the strict
`json_schema` mechanism the Anthropic backend uses is unavailable here.

`tools` and `tool_choice` ARE supported, so the schema is enforced the other
way round: DECISION_SCHEMA is installed as the parameter schema of a single
function, and `tool_choice` forces the model to call it. The model therefore
has to emit arguments in the shape of the schema.

Validation is NOT weakened to accommodate this. Whatever comes back still goes
through the frozen `_coerce`, which is the only authority on what a valid B3
decision is -- unknown action, non-integer delay, negative delay and
out-of-range confidence are all still rejected there. Forced tool calling makes
well-formed output *likelier*; it is not trusted to make it *certain*.

`seed` and `temperature=0` are both supported and are set, which gives this
backend as much determinism as a hosted model can offer. It is still not
byte-reproducible the way MockLLMClient is, and the README must keep saying so.

NO NEW DEPENDENCY
-----------------
OpenRouter is an HTTP endpoint, so this uses `urllib.request` from the standard
library. Nothing is added to requirements.txt, and a clone with no extra
packages still runs the mock benchmark.

THE KEY
-------
Read from OPENROUTER_API_KEY at construction time. It is never logged, never
included in an exception message, never written to the audit trail, and never
returned. A missing key raises immediately, before any evaluation starts, so a
keyless run cannot quietly become a rules-only run wearing a model's name.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional

from ..schema import ABANDON
from .retry import MAX_ATTEMPTS, with_retry
from .llm_client import (
    DECISION_SCHEMA,
    SYSTEM_PROMPT,
    LLMClient,
    LLMDecision,
    _coerce,
)

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
API_KEY_ENV = "OPENROUTER_API_KEY"

# The function the model is forced to call. Its parameter schema IS the frozen
# DECISION_SCHEMA -- not a copy, not a relaxed variant -- so this backend
# cannot drift from the contract the other backends are held to.
TOOL_NAME = "record_recovery_decision"
DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Record the single best next action for this failed "
                       "recurring payment.",
        "parameters": DECISION_SCHEMA,
    },
}

# Deterministic as far as a hosted model allows. Not a reproducibility claim.
SEED = 7


class MissingAPIKey(RuntimeError):
    """Raised when OPENROUTER_API_KEY is absent. Never contains a key."""


class OpenRouterNemotronClient(LLMClient):
    """Calls NVIDIA Nemotron 3.5 Lightning through OpenRouter."""

    name = "nemotron"

    def __init__(self, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 timeout: float = 180.0,
                 attempts: int = MAX_ATTEMPTS):
        super().__init__()
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingAPIKey(
                f"{API_KEY_ENV} is not set. Export it in your shell "
                f"(`export {API_KEY_ENV}=...`) and re-run, or use "
                f"`--llm mock` to run B3 offline. This backend never falls "
                f"back to the mock client, because a rules-only run reported "
                f"as a model result would be a false benchmark."
            )
        self._key = key            # never logged, never surfaced
        self.model = model
        self.timeout = timeout
        self.errors = 0
        # Transient attempts that were retried. Reported for transparency only;
        # it never feeds cost, node counts or recoveries.
        self.retries = 0
        self.attempts = attempts

    # ---------------- transport ----------------

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers. Deliberately generic: no
                # real organisation is named or impersonated.
                "X-Title": "RecoverAI benchmark",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    # ---------------- decision ----------------

    def _decide(self, payload: Dict) -> LLMDecision:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": "Failed recurring payment:\n"
                            + json.dumps(payload, indent=2, sort_keys=True)},
            ],
            "tools": [DECISION_TOOL],
            "tool_choice": {"type": "function",
                            "function": {"name": TOOL_NAME}},
            "temperature": 0,
            "seed": SEED,
            "max_tokens": 2048,
            # Nemotron 3.5 Lightning is a reasoning model, and left to itself it
            # spends its budget thinking: 292 reasoning tokens on a trivial
            # probe, 9-167s per real decision, and occasionally
            # finish_reason=length before answering at all. `reasoning` is in
            # this model's supported_parameters, and disabling it cut a real
            # forced-tool-call decision from 9-167s to ~4.5s with 0 reasoning
            # tokens, while still producing the salary-cycle reasoning the
            # prompt teaches.
            #
            # This IS a material choice about how the model was benchmarked, not
            # an implementation detail: a reasoning model with reasoning off may
            # decide less well, and B3's whole finding is about decision
            # quality. Any published Nemotron result must say so.
            "reasoning": {"enabled": False},
        }

        def _note(attempt, exc, delay):
            self.retries += 1

        try:
            # Only transient provider failures are retried; a 4xx, a malformed
            # answer or a refusal re-raises on the first attempt.
            data = with_retry(lambda: self._post(body),
                              attempts=self.attempts, on_retry=_note)
        except urllib.error.HTTPError as e:
            # Body can carry the provider's reason; the key is only ever in the
            # request header, never in the response, so this is safe to record.
            try:
                detail = e.read().decode()[:200]
            except Exception:
                detail = ""
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0,
                               f"HTTP {e.code}: {detail}"[:300],
                               self.model, malformed=True)
        except Exception as e:
            # Broad on purpose: the goal is not to classify the failure but to
            # guarantee it is COUNTED, so scripts.run_all rejects the whole run
            # rather than publishing a number that is partly the rules baseline.
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0,
                               f"{type(e).__name__}: {e}"[:300],
                               self.model, malformed=True)

        raw = self._extract(data)
        if raw is None:
            self.errors += 1
            snippet = json.dumps(data)[:200]
            return LLMDecision(ABANDON, 0, 0.0,
                               f"no usable tool call in response: {snippet}",
                               self.model, malformed=True)

        # The frozen coercion is the final authority, exactly as for every
        # other backend. Forced tool calling does not earn a shortcut here.
        return _coerce(raw, self.model)

    @staticmethod
    def _extract(data: dict) -> Optional[dict]:
        """
        Pull the decision object out of an OpenAI-shaped response.

        `arguments` is a JSON-encoded string in every OpenAI-compatible
        implementation, but OpenRouter's own documentation does not state it,
        so both a string and an already-decoded object are accepted. A model
        that ignores the forced tool and answers in prose falls through to the
        content field, and if that is not valid JSON the caller records a
        malformed response.
        """
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None

        for call in (msg.get("tool_calls") or []):
            args = (call.get("function") or {}).get("arguments")
            if isinstance(args, dict):
                return args
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if text.startswith("```"):        # fenced JSON is common
                text = text.strip("`")
                text = text.split("\n", 1)[-1] if "\n" in text else text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None
