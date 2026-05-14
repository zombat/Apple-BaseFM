from __future__ import annotations

import dspy
import pytest
from dspy.utils.dummies import DummyLM


# ---------------------------------------------------------------------------
# Simple tool definitions
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


def lookup_capital(country: str) -> str:
    """Return the capital city of the given country."""
    data = {
        "france": "Paris",
        "japan": "Tokyo",
        "germany": "Berlin",
    }
    return data.get(country.lower(), "Unknown")


def always_fails(x: int) -> int:
    """A tool that always raises an exception (for error-handling tests)."""
    raise ValueError(f"Tool failed for input {x}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReActWithTools:
    """dspy.ReAct integration tests using DummyLM (no real LM required)."""

    def test_single_tool_call_and_finish(self) -> None:
        """ReAct calls the add tool once, then finishes with the result."""
        lm = DummyLM(
            [
                # React step 0: decide to call 'add'
                {
                    "next_thought": "I should add 3 and 4.",
                    "next_tool_name": "add",
                    "next_tool_args": {"a": 3, "b": 4},
                },
                # React step 1: call finish
                {
                    "next_thought": "I have the sum, I can finish now.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                # ChainOfThought extract step
                {"reasoning": "3 + 4 = 7", "answer": "7"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[add])
            result = react(question="What is 3 + 4?")

        assert result.answer == "7"
        traj = result.trajectory
        assert traj["thought_0"] == "I should add 3 and 4."
        assert traj["tool_name_0"] == "add"
        assert traj["tool_args_0"] == {"a": 3, "b": 4}
        # The real tool was called — observation is the actual return value
        assert traj["observation_0"] == 7

    def test_tool_result_stored_in_trajectory(self) -> None:
        """The observation recorded in the trajectory matches the real tool return value."""
        lm = DummyLM(
            [
                {
                    "next_thought": "I'll multiply 6 by 7.",
                    "next_tool_name": "multiply",
                    "next_tool_args": {"a": 6, "b": 7},
                },
                {
                    "next_thought": "Done.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "6 * 7 = 42", "answer": "42"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[multiply])
            result = react(question="What is 6 * 7?")

        assert result.trajectory["observation_0"] == 42

    def test_lookup_tool(self) -> None:
        """ReAct uses a string-returning lookup tool to answer a factual question."""
        lm = DummyLM(
            [
                {
                    "next_thought": "I need to look up the capital of France.",
                    "next_tool_name": "lookup_capital",
                    "next_tool_args": {"country": "France"},
                },
                {
                    "next_thought": "I have the answer.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "The capital of France is Paris.", "answer": "Paris"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[lookup_capital])
            result = react(question="What is the capital of France?")

        assert result.answer == "Paris"
        assert result.trajectory["observation_0"] == "Paris"

    def test_tool_error_recorded_as_observation(self) -> None:
        """When a tool raises, the error string is stored in the trajectory."""
        lm = DummyLM(
            [
                {
                    "next_thought": "I'll call always_fails.",
                    "next_tool_name": "always_fails",
                    "next_tool_args": {"x": 1},
                },
                {
                    "next_thought": "The tool failed; I'll finish anyway.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "Tool failed but we can still answer.", "answer": "unknown"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[always_fails])
            result = react(question="Try always_fails(1)")

        obs = result.trajectory["observation_0"]
        # ReAct wraps tool exceptions in an "Execution error in <tool>" message
        assert "Execution error" in obs
        assert "always_fails" in obs

    def test_multiple_tools_available_selects_correct_one(self) -> None:
        """ReAct selects the correct tool when several are available."""
        lm = DummyLM(
            [
                {
                    "next_thought": "I should look up the capital of Japan.",
                    "next_tool_name": "lookup_capital",
                    "next_tool_args": {"country": "Japan"},
                },
                {
                    "next_thought": "I have the capital.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "The capital is Tokyo.", "answer": "Tokyo"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[add, multiply, lookup_capital])
            result = react(question="What is the capital of Japan?")

        assert result.answer == "Tokyo"
        assert result.trajectory["tool_name_0"] == "lookup_capital"

    def test_direct_finish_no_tool_call(self) -> None:
        """ReAct can finish on the first step without calling any tool."""
        lm = DummyLM(
            [
                {
                    "next_thought": "I already know the answer.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "The sky is blue.", "answer": "blue"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[add])
            result = react(question="What color is the sky?")

        assert result.answer == "blue"
        assert result.trajectory["tool_name_0"] == "finish"

    def test_multi_step_chained_tools(self) -> None:
        """ReAct calls multiple tools in sequence before finishing."""
        lm = DummyLM(
            [
                # Step 0: add 2 + 3
                {
                    "next_thought": "First I'll add 2 and 3.",
                    "next_tool_name": "add",
                    "next_tool_args": {"a": 2, "b": 3},
                },
                # Step 1: multiply result by 4
                {
                    "next_thought": "Now I'll multiply 5 by 4.",
                    "next_tool_name": "multiply",
                    "next_tool_args": {"a": 5, "b": 4},
                },
                # Step 2: finish
                {
                    "next_thought": "I have the final result: 20.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                {"reasoning": "(2+3)*4 = 20", "answer": "20"},
            ]
        )
        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[add, multiply])
            result = react(question="What is (2+3)*4?")

        assert result.answer == "20"
        traj = result.trajectory
        assert traj["observation_0"] == 5    # add(2, 3)
        assert traj["observation_1"] == 20   # multiply(5, 4)
        assert traj["tool_name_0"] == "add"
        assert traj["tool_name_1"] == "multiply"
