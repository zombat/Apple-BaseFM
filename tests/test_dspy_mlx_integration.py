from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import patch

import dspy
import pytest
from dspy.utils.dummies import DummyLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dspy_fmt(fields: dict) -> str:
    """Return a ChatAdapter-formatted response string for the given field dict.

    Delegates to DummyLM's internal formatter so the output is always in the
    exact format the installed DSPy version expects — no hard-coded marker strings.
    """
    dummy = DummyLM([fields])
    response = dummy.forward(messages=[{"role": "user", "content": "x"}])
    return response.choices[0].message.content


def _reload_apple_local() -> types.ModuleType:
    for key in list(sys.modules):
        if "apple_basefm.apple_local" in key:
            del sys.modules[key]
    return importlib.import_module("apple_basefm.apple_local")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def apple_local_mod(fake_mlx_modules: types.ModuleType) -> types.ModuleType:  # noqa: ARG001
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return _reload_apple_local()


@pytest.fixture()
def lm(apple_local_mod: types.ModuleType) -> Any:
    """AppleLocalLM backed by fake mlx_lm; caching disabled for deterministic sequencing."""
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return apple_local_mod.AppleLocalLM("mlx-community/test-model-4bit", cache=False)


def _seq_generate(instance: Any, responses: list[str]):
    """Context manager: replace _generate with an iterator over *responses*.

    Each call to _generate consumes the next string from the list.
    The fake always returns ("response_text", "fake-flat-prompt").
    """
    itr = iter(responses)

    def _fake(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        return next(itr, "[[## answer ##]]\
fallback\
"), "fake-prompt"

    return patch.object(instance, "_generate", side_effect=_fake)


# ---------------------------------------------------------------------------
# Tool definitions used by ReAct tests
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def lookup_capital(country: str) -> str:
    """Return the capital city of the given country."""
    return {"france": "Paris", "germany": "Berlin"}.get(country.lower(), "unknown")


# ---------------------------------------------------------------------------
# dspy.Predict
# ---------------------------------------------------------------------------


class TestPredictWithMLX:
    """dspy.Predict driven by AppleLocalLM."""

    def test_single_output_field(self, lm: Any) -> None:
        """Predict extracts a single output field from the MLX response."""
        with _seq_generate(lm, [_dspy_fmt({"answer": "Paris"})]):
            with dspy.context(lm=lm):
                result = dspy.Predict("question -> answer")(question="Capital of France?")
        assert result.answer.strip() == "Paris"

    def test_multiple_output_fields(self, lm: Any) -> None:
        """Predict extracts multiple output fields from a single MLX response."""
        with _seq_generate(lm, [_dspy_fmt({"capital": "Paris", "country": "France"})]):
            with dspy.context(lm=lm):
                result = dspy.Predict("question -> capital, country")(
                    question="What is France's capital and name?"
                )
        assert result.capital.strip() == "Paris"
        assert result.country.strip() == "France"

    def test_forward_receives_messages_kwarg(self, lm: Any) -> None:
        """DSPy calls forward() with a messages= kwarg (OpenAI chat format contract)."""
        forward_calls: list[dict] = []
        original = lm.forward

        def _spy(**kwargs: Any) -> Any:
            forward_calls.append(kwargs)
            return original(**kwargs)

        with patch.object(lm, "forward", wraps=_spy):
            with _seq_generate(lm, [_dspy_fmt({"answer": "Berlin"})]):
                with dspy.context(lm=lm):
                    dspy.Predict("question -> answer")(question="Capital of Germany?")

        assert len(forward_calls) >= 1
        assert "messages" in forward_calls[0]
        assert isinstance(forward_calls[0]["messages"], list)

    def test_direct_lm_call_returns_list_of_strings(self, lm: Any) -> None:
        """BaseLM.__call__(messages=[...]) returns list[str] — the DSPy LM contract."""
        with _seq_generate(lm, [_dspy_fmt({"answer": "Paris"})]):
            result = lm(messages=[{"role": "user", "content": "Capital of France?"}])
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], str)

    def test_response_cost_is_zero_in_hidden_params(self, lm: Any) -> None:
        """_FMResponse._hidden_params must carry response_cost=0.0 (DSPy history contract)."""
        responses: list[Any] = []
        original = lm.forward

        def _capture(**kwargs: Any) -> Any:
            r = original(**kwargs)
            responses.append(r)
            return r

        with patch.object(lm, "forward", wraps=_capture):
            with _seq_generate(lm, [_dspy_fmt({"answer": "Paris"})]):
                lm(messages=[{"role": "user", "content": "test"}])

        assert responses[0]._hidden_params.get("response_cost") == 0.0


# ---------------------------------------------------------------------------
# dspy.ChainOfThought
# ---------------------------------------------------------------------------


class TestChainOfThoughtWithMLX:
    """dspy.ChainOfThought driven by AppleLocalLM."""

    def test_answer_extracted(self, lm: Any) -> None:
        """ChainOfThought extracts the answer field from the MLX response."""
        with _seq_generate(
            lm, [_dspy_fmt({"reasoning": "France's capital is Paris.", "answer": "Paris"})]
        ):
            with dspy.context(lm=lm):
                result = dspy.ChainOfThought("question -> answer")(
                    question="Capital of France?"
                )
        assert result.answer.strip() == "Paris"

    def test_reasoning_field_populated(self, lm: Any) -> None:
        """ChainOfThought result has a non-empty reasoning attribute."""
        with _seq_generate(
            lm, [_dspy_fmt({"reasoning": "Step-by-step reasoning.", "answer": "Paris"})]
        ):
            with dspy.context(lm=lm):
                result = dspy.ChainOfThought("question -> answer")(
                    question="Capital of France?"
                )
        assert hasattr(result, "reasoning")
        assert result.reasoning is not None
        assert result.reasoning.strip() != ""

    def test_multiple_output_fields(self, lm: Any) -> None:
        """ChainOfThought extracts multiple output fields alongside reasoning."""
        with _seq_generate(
            lm,
            [
                _dspy_fmt(
                    {
                        "reasoning": "Analysed both fields.",
                        "name": "Paris",
                        "population": "2.1 million",
                    }
                )
            ],
        ):
            with dspy.context(lm=lm):
                result = dspy.ChainOfThought("question -> name, population")(
                    question="Tell me about France's capital."
                )
        assert result.name.strip() == "Paris"
        assert result.population.strip() == "2.1 million"


# ---------------------------------------------------------------------------
# dspy.ReAct
# ---------------------------------------------------------------------------


class TestReActWithMLX:
    """dspy.ReAct driven by AppleLocalLM."""

    def test_immediate_finish_returns_answer(self, lm: Any) -> None:
        """ReAct returns the correct answer when the model finishes on the first step."""
        react_resp = _dspy_fmt(
            {"next_thought": "I know.", "next_tool_name": "finish", "next_tool_args": {}}
        )
        extract_resp = _dspy_fmt({"reasoning": "Already known.", "answer": "Paris"})
        with _seq_generate(lm, [react_resp, extract_resp]):
            with dspy.context(lm=lm):
                result = dspy.ReAct("question -> answer", tools=[lookup_capital])(
                    question="Capital of France?"
                )
        assert result.answer.strip() == "Paris"

    def test_tool_call_observation_recorded_in_trajectory(self, lm: Any) -> None:
        """ReAct calls a tool; the real return value is stored in trajectory observation."""
        tool_call_resp = _dspy_fmt(
            {
                "next_thought": "I need to add 3 and 4.",
                "next_tool_name": "add",
                "next_tool_args": {"a": 3, "b": 4},
            }
        )
        finish_resp = _dspy_fmt(
            {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}}
        )
        extract_resp = _dspy_fmt({"reasoning": "3+4=7.", "answer": "7"})
        with _seq_generate(lm, [tool_call_resp, finish_resp, extract_resp]):
            with dspy.context(lm=lm):
                result = dspy.ReAct("question -> answer", tools=[add])(
                    question="What is 3 + 4?"
                )
        # The real add() function is called — observation is the actual Python return value.
        assert result.trajectory["observation_0"] == 7
        assert result.answer.strip() == "7"

    def test_trajectory_keys_present(self, lm: Any) -> None:
        """ReAct trajectory dict contains thought/tool_name/tool_args/observation for each step."""
        react_resp = _dspy_fmt(
            {"next_thought": "I know.", "next_tool_name": "finish", "next_tool_args": {}}
        )
        extract_resp = _dspy_fmt({"reasoning": "Known.", "answer": "Paris"})
        with _seq_generate(lm, [react_resp, extract_resp]):
            with dspy.context(lm=lm):
                result = dspy.ReAct("question -> answer", tools=[lookup_capital])(
                    question="Capital of France?"
                )
        traj = result.trajectory
        assert "thought_0" in traj
        assert "tool_name_0" in traj
        assert "tool_args_0" in traj
        assert "observation_0" in traj

    def test_tool_error_stored_as_observation(self, lm: Any) -> None:
        """When a tool raises, ReAct stores the error string and continues."""

        def broken_tool(x: int) -> int:
            """A tool that always fails."""
            raise ValueError(f"tool failed on input {x}")

        tool_call_resp = _dspy_fmt(
            {
                "next_thought": "I'll call broken_tool.",
                "next_tool_name": "broken_tool",
                "next_tool_args": {"x": 1},
            }
        )
        finish_resp = _dspy_fmt(
            {"next_thought": "Tool failed; finishing.", "next_tool_name": "finish", "next_tool_args": {}}
        )
        extract_resp = _dspy_fmt({"reasoning": "Tool failed.", "answer": "unknown"})
        with _seq_generate(lm, [tool_call_resp, finish_resp, extract_resp]):
            with dspy.context(lm=lm):
                result = dspy.ReAct("question -> answer", tools=[broken_tool])(
                    question="Call broken_tool(1)."
                )
        obs = result.trajectory["observation_0"]
        assert "Execution error" in obs
        assert "broken_tool" in obs


# ---------------------------------------------------------------------------
# Custom dspy.Module pipelines
# ---------------------------------------------------------------------------


class TestCustomModuleWithMLX:
    """Multi-step dspy.Module pipelines driven by AppleLocalLM."""

    def test_two_stage_predict_pipeline(self, lm: Any) -> None:
        """A two-step Module (extract → reason) routes both calls through MLX."""

        class ExtractThenReason(dspy.Module):
            def __init__(self) -> None:
                self.extract = dspy.Predict("raw_text -> entities")
                self.reason = dspy.Predict("entities -> verdict")

            def forward(self, raw_text: str) -> dspy.Prediction:
                extracted = self.extract(raw_text=raw_text)
                return self.reason(entities=extracted.entities)

        extract_resp = _dspy_fmt({"entities": "Apple, M4 chip, 2024"})
        reason_resp = _dspy_fmt({"verdict": "positive announcement"})
        with _seq_generate(lm, [extract_resp, reason_resp]):
            with dspy.context(lm=lm):
                result = ExtractThenReason().forward(
                    raw_text="Apple announced M4 chip in 2024."
                )
        assert result.verdict.strip() == "positive announcement"

    def test_cot_plus_predict_pipeline(self, lm: Any) -> None:
        """A module mixing ChainOfThought and Predict uses the same MLX LM for both."""

        class HybridModule(dspy.Module):
            def __init__(self) -> None:
                self.think = dspy.ChainOfThought("input -> analysis")
                self.decide = dspy.Predict("analysis -> decision")

            def forward(self, input: str) -> dspy.Prediction:
                thought = self.think(input=input)
                return self.decide(analysis=thought.analysis)

        cot_resp = _dspy_fmt({"reasoning": "Breaking it down.", "analysis": "Positive data."})
        decide_resp = _dspy_fmt({"decision": "approved"})
        with _seq_generate(lm, [cot_resp, decide_resp]):
            with dspy.context(lm=lm):
                result = HybridModule().forward(input="Review quarterly data.")
        assert result.decision.strip() == "approved"


# ---------------------------------------------------------------------------
# dspy.context — LM scope isolation
# ---------------------------------------------------------------------------


class TestDSPyContextWithMLX:
    """dspy.context(lm=lm) correctly scopes and restores the configured LM."""

    def test_context_sets_lm_inside_block(self, lm: Any) -> None:
        """Inside dspy.context(lm=lm), dspy.settings.lm is the AppleLocalLM instance."""
        with dspy.context(lm=lm):
            assert dspy.settings.lm is lm

    def test_context_restores_outer_lm_on_exit(self, lm: Any) -> None:
        """After exiting dspy.context, the previously configured LM is restored."""
        outer = dspy.settings.lm
        with dspy.context(lm=lm):
            pass
        assert dspy.settings.lm is outer

    def test_nested_context_restores_intermediate_lm(
        self, lm: Any, apple_local_mod: types.ModuleType
    ) -> None:
        """Nested dspy.context scopes restore the correct LM at each level."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            lm2 = apple_local_mod.AppleLocalLM("mlx-community/other-model", cache=False)

        with dspy.context(lm=lm):
            assert dspy.settings.lm is lm
            with dspy.context(lm=lm2):
                assert dspy.settings.lm is lm2
            assert dspy.settings.lm is lm


# ---------------------------------------------------------------------------
# token_session() + DSPy modules
# ---------------------------------------------------------------------------


class TestTokenSessionWithMLX:
    """token_session() accumulates tokens/calls when DSPy modules drive AppleLocalLM."""

    def test_single_predict_increments_call_count(self, lm: Any) -> None:
        """One dspy.Predict call inside a session increments call_count by 1."""
        from apple_basefm import token_session

        with _seq_generate(lm, [_dspy_fmt({"answer": "Paris"})]):
            with dspy.context(lm=lm):
                with token_session() as session:
                    dspy.Predict("question -> answer")(question="Capital of France?")

        assert session.call_count == 1

    def test_multiple_predict_calls_accumulate_count(self, lm: Any) -> None:
        """Multiple Predict calls within one session accumulate call_count correctly."""
        from apple_basefm import token_session

        responses = [
            _dspy_fmt({"answer": "Paris"}),
            _dspy_fmt({"answer": "Berlin"}),
            _dspy_fmt({"answer": "Tokyo"}),
        ]
        with _seq_generate(lm, responses):
            with dspy.context(lm=lm):
                qa = dspy.Predict("question -> answer")
                with token_session() as session:
                    qa(question="Capital of France?")
                    qa(question="Capital of Germany?")
                    qa(question="Capital of Japan?")

        assert session.call_count == 3

    def test_call_outside_session_not_counted(self, lm: Any) -> None:
        """A call made after the session exits does not increment the closed session."""
        from apple_basefm import token_session

        resp = _dspy_fmt({"answer": "Paris"})
        qa = dspy.Predict("question -> answer")
        with _seq_generate(lm, [resp, resp]):
            with dspy.context(lm=lm):
                with token_session() as session:
                    qa(question="First call — inside session.")
                # Session has exited; this call must NOT be counted.
                qa(question="Second call — outside session.")

        assert session.call_count == 1

    def test_nested_sessions_isolated(self, lm: Any) -> None:
        """Inner token_session counts only its own calls; outer resumes after inner exits."""
        from apple_basefm import token_session

        responses = [
            _dspy_fmt({"answer": "outer-1"}),
            _dspy_fmt({"answer": "inner"}),
            _dspy_fmt({"answer": "outer-2"}),
        ]
        qa = dspy.Predict("question -> answer")
        with _seq_generate(lm, responses):
            with dspy.context(lm=lm):
                with token_session() as outer:
                    qa(question="outer q1")
                    with token_session() as inner:
                        qa(question="inner q")
                    qa(question="outer q2")

        assert inner.call_count == 1
        assert outer.call_count == 2

    def test_session_token_counts_are_non_negative(self, lm: Any) -> None:
        """Token counts (prompt, completion, total) are non-negative integers after a call."""
        from apple_basefm import token_session

        with _seq_generate(lm, [_dspy_fmt({"answer": "Paris"})]):
            with dspy.context(lm=lm):
                with token_session() as session:
                    dspy.Predict("question -> answer")(question="Capital of France?")

        assert session.prompt_tokens >= 0
        assert session.completion_tokens >= 0
        assert session.total_tokens == session.prompt_tokens + session.completion_tokens
        assert session.call_count == 1
