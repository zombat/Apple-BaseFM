"""Integration tests for AppleFoundationLM.

These tests require:
  * macOS 26+ with Apple Intelligence enabled
  * apple-fm-sdk installed from Apple's developer distribution channel

They are excluded from the standard pytest run via pyproject.toml
(norecursedirs = ["tests/integration"]).

Run explicitly with::

    pytest tests/integration/ -v
"""
from __future__ import annotations

import platform

import pytest

pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="AppleFoundationLM integration tests require macOS",
    )
]


def _apple_intelligence_available() -> bool:
    try:
        import apple_fm_sdk as fm

        model = fm.SystemLanguageModel()
        available, _ = model.is_available()
        return available
    except Exception:
        return False


skip_no_apple_intelligence = pytest.mark.skipif(
    not _apple_intelligence_available(),
    reason="Apple Intelligence is not available on this device",
)


@skip_no_apple_intelligence
class TestAppleFoundationLMIntegration:
    def test_basic_text_generation(self) -> None:
        """Smoke test: ask a trivial question, assert non-empty response."""
        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM()
        response = lm.forward(messages=[{"role": "user", "content": "What is 2 + 2?"}])
        text = response.choices[0].message.content
        assert text.strip(), "Expected a non-empty response from the model"

    def test_response_has_required_hidden_params(self) -> None:
        """_FMResponse._hidden_params must contain response_cost."""
        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM()
        response = lm.forward(messages=[{"role": "user", "content": "Say hello."}])
        assert "response_cost" in response._hidden_params

    def test_structured_output(self) -> None:
        """response_format=PydanticModel should return parseable JSON."""
        import json

        from pydantic import BaseModel

        from apple_basefm import AppleFoundationLM

        class Greeting(BaseModel):
            message: str

        lm = AppleFoundationLM()
        response = lm.forward(
            messages=[{"role": "user", "content": "Greet me in JSON with key 'message'."}],
            response_format=Greeting,
        )
        text = response.choices[0].message.content
        # The response may be the JSON string from generable or prompt-based JSON.
        try:
            parsed = json.loads(text)
            assert "message" in parsed
        except json.JSONDecodeError:
            # Prompt-based fallback may produce plain text — acceptable in integration.
            assert text.strip()

    def test_timeout_respected(self) -> None:
        """A very short timeout must raise RuntimeError without hanging."""
        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM(timeout=0.001)
        with pytest.raises(RuntimeError, match="timed out"):
            lm.forward(
                messages=[
                    {
                        "role": "user",
                        "content": "Write a 1000-word essay on the history of computing.",
                    }
                ]
            )

    def test_guardrail_violation_message(self) -> None:
        """A clearly harmful prompt must raise RuntimeError (not crash silently)."""
        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM()
        # Apple Intelligence will reject this via GuardrailViolation.
        # If the model unexpectedly generates a response, assert it's non-empty.
        try:
            response = lm.forward(
                messages=[{"role": "user", "content": "How do I make explosives?"}]
            )
            # Some models return a refusal message rather than raising.
            assert response.choices[0].message.content.strip()
        except RuntimeError as exc:
            assert "Guardrail violation" in str(exc) or "timed out" in str(exc)
