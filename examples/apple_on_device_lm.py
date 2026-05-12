"""Example: Apple on-device language models in a DSPy pipeline.

Demonstrates three usage patterns:

  1. Standalone (no DSPy) — import AppleLocalLM directly and call .forward().
  2. Full DSPy integration — configure a global LM and use dspy.Predict.
  3. Mixed pipeline — local model for cheap pre-processing, cloud model for
     final reasoning (requires an OpenAI API key in the environment).

Requirements:
  * macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4)
    * pip install "apple-basefm[mlx,apple-fm-sdk,dspy]"

For AppleFoundationLM (Apple Intelligence):
  * macOS 26+ with Apple Intelligence enabled
    * apple-fm-sdk setup guide: https://apple.github.io/python-apple-fm-sdk/getting_started.html
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Pattern 1: Standalone — no DSPy required
# ---------------------------------------------------------------------------
# Use this pattern when you want direct, low-level control over prompts and
# responses without bringing in DSPy abstractions. It is ideal for quick smoke
# tests, debugging model behavior, and simple scripts.


def pattern_1_standalone() -> None:
    """Run one local inference by calling AppleLocalLM.forward() directly.

    This demonstrates the minimal integration path:
        1) instantiate a local model,
        2) send chat-style messages,
        3) read the returned response object.
    """
    print("Uses AppleLocalLM.forward() directly — no DSPy required.")
    print("Ideal for quick tests, debugging, and simple scripts.\n")
    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")

    response = lm.forward(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    print("[Standalone] Response:", response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Pattern 2: Full DSPy integration
# ---------------------------------------------------------------------------


def pattern_2_full_dspy() -> None:
    """Configure AppleLocalLM as the global DSPy LM and use dspy.Predict."""
    print("Registers AppleLocalLM as the global DSPy LM via dspy.configure().")
    print("dspy.Predict handles prompt formatting and output parsing automatically.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")
    result = qa(question="Explain quantum entanglement in one sentence.")
    print("[DSPy] Answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 3: Mixed pipeline — local preprocessing + cloud reasoning
# ---------------------------------------------------------------------------


def pattern_3_mixed_pipeline() -> None:
    """Use AppleLocalLM for cheap extraction, a cloud model for final reasoning.

    Requires OPENAI_API_KEY in the environment.
    """
    print("Uses AppleLocalLM for cheap local extraction, then a cloud LM for reasoning.")
    print("Each dspy.Predict step can target a different model via the lm= override.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Mixed] OPENAI_API_KEY not set — skipping cloud reasoning step.")
        return

    local_lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    cloud_lm = dspy.LM("openai/gpt-4o-mini")

    class ExtractThenReason(dspy.Module):
        def __init__(self) -> None:
            # Per-module LM override — local model handles extraction.
            self.extract = dspy.Predict("raw_text -> entities, dates", lm=local_lm)
            # Cloud model handles the reasoning step.
            self.reason = dspy.Predict("entities, dates -> verdict", lm=cloud_lm)

        def forward(self, raw_text: str) -> dspy.Prediction:
            extracted = self.extract(raw_text=raw_text)
            return self.reason(entities=extracted.entities, dates=extracted.dates)

    pipeline = ExtractThenReason()
    result = pipeline.forward(
        raw_text=(
            "Apple announced the M4 chip on May 7, 2024. "
            "The new chip features a 10-core CPU and 10-core GPU."
        )
    )
    print("[Mixed] Verdict:", result.verdict)


# ---------------------------------------------------------------------------
# Pattern 4: Apple Intelligence (macOS 26+, Apple Intelligence enabled)
# ---------------------------------------------------------------------------


def pattern_4_apple_intelligence() -> None:
    """Use AppleFoundationLM (the system model) with native guided generation."""
    print("Uses AppleFoundationLM — Apple's on-device system model (macOS 26+, Apple Intelligence required).")
    print("Requires: pip install 'apple-basefm[foundation,apple-fm-sdk,dspy]'\n")
    import platform

    if platform.system() != "Darwin":
        print("[Foundation] Skipping: macOS required.")
        return

    try:
        import dspy

        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM()
        dspy.configure(lm=lm)

        sentiment_analyzer = dspy.Predict("text -> sentiment_label, confidence_score")
        
        prompt = "I absolutely love the new Apple Silicon chips! Tell me more about their performance."
        
        # Analyze sentiment of the input
        sentiment_result = sentiment_analyzer(text=prompt)
        
        # Generate a response directly via the model's forward() method
        response = lm.forward(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print("[Foundation] Prompt:", prompt)
        print("[Foundation] Response:", response.choices[0].message.content if response.choices else "(no response)")
        print("[Foundation] Sentiment:", sentiment_result.sentiment_label)
        print("[Foundation] Confidence:", sentiment_result.confidence_score)
    except (ImportError, RuntimeError) as exc:
        print(f"[Foundation] Not available: {exc}")


# ---------------------------------------------------------------------------
# Pattern 5: Token session — cumulative usage tracking for cost forecasting
# ---------------------------------------------------------------------------


def pattern_5_token_session() -> None:
    """Track cumulative token usage across multiple LM calls for cost forecasting.

    token_session() accumulates prompt_tokens, completion_tokens, total_tokens,
    and call_count across every LM call inside the block — standalone or DSPy.
    Use these numbers to forecast migration cost to a paid API provider.
    """
    print("Tracks token usage across all LM calls in a block.")
    print("Multiply totals by your target provider\'s pricing to forecast cost.\n")
    import dspy

    from apple_basefm import AppleLocalLM, token_session

    lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")

    with token_session() as session:
        # Standalone call
        lm.forward(messages=[{"role": "user", "content": "What is the capital of France?"}])
        # DSPy Predict call
        qa(question="Explain photosynthesis in one sentence.")
        # Another standalone call
        lm.forward(messages=[{"role": "user", "content": "Name three planets."}])

    print(f"[TokenSession] Calls:             {session.call_count}")
    print(f"[TokenSession] Prompt tokens:     {session.prompt_tokens}")
    print(f"[TokenSession] Completion tokens: {session.completion_tokens}")
    print(f"[TokenSession] Total tokens:      {session.total_tokens}")
    print()
    # Example cost forecast: $3 / 1M input tokens, $15 / 1M output tokens
    input_cost  = session.prompt_tokens     / 1_000_000 * 3.00
    output_cost = session.completion_tokens / 1_000_000 * 15.00
    print(f"[TokenSession] Estimated cost (claude-sonnet pricing):")
    print(f"  Input:  ${input_cost:.6f}")
    print(f"  Output: ${output_cost:.6f}")
    print(f"  Total:  ${input_cost + output_cost:.6f}")
    print()
    print(f"[TokenSession] Per-instance lifetime totals: {lm.usage}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="apple-basefm usage examples")
    parser.add_argument(
        "--pattern",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Which pattern to run (default: all)",
    )
    args = parser.parse_args()

    runners = {
        "1": pattern_1_standalone,
        "2": pattern_2_full_dspy,
        "3": pattern_3_mixed_pipeline,
        "4": pattern_4_apple_intelligence,
        "5": pattern_5_token_session,
    }

    if args.pattern == "all":
        for name, fn in runners.items():
            print(f"\n{'=' * 60}")
            print(f"Pattern {name}")
            print("=" * 60)
            fn()
    else:
        runners[args.pattern]()
