"""
Gemini backend for B3, as an alternative to the Anthropic one.

    pip install google-genai
    export GEMINI_API_KEY=...
    python -m scripts.run_all --split test --llm gemini

WHY THIS IS A SEPARATE MODULE
-----------------------------
`llm_client.py` is frozen: it holds MockLLMClient, the prompt, the response
schema and the response coercion that every backend shares, and the benchmark's
reproducibility claim rests on it not moving. Adding a provider by subclassing
from the outside touches none of it.

APPLES TO APPLES
----------------
This client deliberately reuses `SYSTEM_PROMPT`, `DECISION_SCHEMA` and
`_coerce` from llm_client rather than writing its own. Same instructions, same
output contract, same validation -- so a Gemini result and an Anthropic result
differ only by the model, which is the only thing the comparison is about. A
provider-specific prompt would quietly turn a model comparison into a prompt
comparison.

WHAT THIS BACKEND DOES NOT CHANGE
---------------------------------
Nothing about the benchmark. Routing still depends only on B2's expected-value
margins, which are model-independent, so the same 262 records are routed. The
schema, retry-budget, compliance and economic gates are unchanged, and a model
recommendation still cannot execute. Every failure increments `.errors`, which
makes `scripts.run_all` reject the whole run rather than publish a figure that
is partly the rules baseline wearing a model's name.
"""

from __future__ import annotations

import json
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

# Stable Flash model, described by Google as their current Flash generation for
# agentic work. `gemini-3.1-pro-preview` is more capable but is a preview; pass
# --model to use it if you would rather benchmark the top of the range.
DEFAULT_MODEL = "gemini-3.8-flash"

# Deterministic as far as a hosted model allows. Not a reproducibility claim.
SEED = 7

# The shared schema minus `additionalProperties`, which is an Anthropic
# strict-mode requirement rather than part of the contract. The property set,
# the enum of valid actions and the required fields are all inherited, so both
# backends are held to the same output shape.
_GEMINI_SCHEMA = {k: v for k, v in DECISION_SCHEMA.items()
                  if k != "additionalProperties"}


class GeminiLLMClient(LLMClient):
    """Calls Gemini for a single recovery decision."""

    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 attempts: int = MAX_ATTEMPTS):
        super().__init__()
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "The google-genai SDK is not installed. `pip install google-genai`, "
                "or use MockLLMClient to run B3 offline."
            ) from e
        self._genai = genai
        self._types = genai_types
        self.model = model
        # A bare Client() resolves GEMINI_API_KEY from the environment, so no
        # key is ever read, stored or logged by this repository.
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.errors = 0
        # Transient attempts that were retried. Reported for transparency only;
        # it never feeds cost, node counts or recoveries.
        self.retries = 0
        self.attempts = attempts

    def _decide(self, payload: Dict) -> LLMDecision:
        def _call():
            return self.client.models.generate_content(
                model=self.model,
                contents=("Failed recurring payment:\n"
                          + json.dumps(payload, indent=2, sort_keys=True)),
                config=self._types.GenerateContentConfig(
                    # The frozen prompt goes in as a system instruction rather
                    # than being glued onto the user turn, matching how the
                    # other backends separate the two.
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    # response_json_schema takes a raw JSON Schema dict;
                    # response_schema expects the SDK's own Schema type.
                    response_json_schema=_GEMINI_SCHEMA,
                    temperature=0,
                    seed=SEED,
                    max_output_tokens=2048,
                ),
            )

        def _note(attempt, exc, delay):
            self.retries += 1

        try:
            # Only transient provider failures (429/503/timeout/reset) are
            # retried, and only up to `attempts` total. A malformed answer, a
            # refusal or a 4xx re-raises immediately -- those are answers, not
            # noise, and must stay fail-loud.
            response = with_retry(_call, attempts=self.attempts, on_retry=_note)
        except Exception as e:
            # Deliberately broad. The point of catching here is not to classify
            # the failure but to guarantee it is COUNTED: any error at all makes
            # run_all reject the run, so a network blip cannot be laundered into
            # a benchmark number. The exception type is recorded so the cause is
            # still visible in the audit trail.
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0,
                               f"{type(e).__name__}: {e}"[:300],
                               self.model, malformed=True)

        text = getattr(response, "text", None)
        if not text:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, "empty response",
                               self.model, malformed=True)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            self.errors += 1
            return LLMDecision(ABANDON, 0, 0.0, "unparseable response",
                               self.model, malformed=True)
        return _coerce(raw, self.model)
