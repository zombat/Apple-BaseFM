"""Example: Apple on-device language models in a DSPy pipeline.

Demonstrates twenty usage patterns:

  1.  Standalone (no DSPy) — import AppleLocalLM directly and call .forward().
  2.  dspy.Predict — configure a global LM and run a simple QA predictor.
  3.  Mixed pipeline — local model for cheap pre-processing, cloud model for
      final reasoning (requires an OpenAI API key in the environment).
  4.  Apple Intelligence — AppleFoundationLM + dspy.Predict (macOS 26+).
  5.  Token session — cumulative usage tracking for cost forecasting.
  6.  dspy.ChainOfThought — structured reasoning + answer extraction.
  7.  dspy.ReAct — agentic tool use with trajectory inspection.
  8.  Custom dspy.Module pipeline — multi-step Predict chain.
  9.  dspy.context — per-call LM scoping without touching global state.
  10. AppleFoundationLM + dspy.ReAct — native tool calling (macOS 26+).
  11. Streaming — token-by-token output via dspy.streamify.
  12. MCP tools — dspy.ReAct with dspy.Tool.from_mcp_tool and a FastMCP server.
  13. Typed Signature — class-based dspy.Signature with InputField/OutputField.
  14. dspy.BestOfN — sample N candidates, return the highest-reward answer.
  15. dspy.Refine — iterative refinement with per-attempt feedback.
  16. dspy.MultiChainComparison — M independent CoT chains, holistic comparison.
  17. dspy.ProgramOfThought — LM writes Python; code executes locally.
  18. dspy.Parallel — concurrent (module, inputs) pairs across a thread pool.
  19. dspy.CodeAct — LM emits Python code snippets; tools run in sandbox.
  20. dspy.RLM — REPL-history code interpreter, reinforcement-learning loop.
  21. Raw mlx-lm forward pass — logits, embeddings, and custom decoding.

Requirements:
  * macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4)
    * pip install "apple-basefm[mlx,apple-fm-sdk,dspy]"

For AppleFoundationLM (Apple Intelligence):
  * macOS 26+ with Apple Intelligence enabled
    * apple-fm-sdk setup guide: https://apple.github.io/python-apple-fm-sdk/getting_started.html

For Pattern 12 (MCP tools):
  * pip install "fastmcp>=2.0"
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

SMALL_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
LARGE_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q4"

# Resolved at startup from --model; patterns reference this name.
_DEFAULT_MODEL = SMALL_MODEL

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

    lm = AppleLocalLM(_DEFAULT_MODEL)

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

    lm = AppleLocalLM(_DEFAULT_MODEL)
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

    local_lm = AppleLocalLM(_DEFAULT_MODEL)
    cloud_lm = dspy.LM("openai/gpt-4o-mini")

    class ExtractThenReason(dspy.Module):
        def __init__(self) -> None:
            self.extract = dspy.Predict("raw_text -> entities, dates")
            self.extract.lm = local_lm  # per-predictor LM override via .lm
            self.reason = dspy.Predict("entities, dates -> verdict")
            self.reason.lm = cloud_lm

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

    # cache=False ensures every call hits the model — token_session only counts
    # real generation paths; DSPy cache hits are intentionally excluded because
    # cached responses have no inference cost to forecast.
    lm = AppleLocalLM(_DEFAULT_MODEL, cache=False)
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
# Pattern 6: dspy.ChainOfThought
# ---------------------------------------------------------------------------


def pattern_6_chain_of_thought() -> None:
    """Use dspy.ChainOfThought to produce a reasoning trace before the answer.

    ChainOfThought automatically prepends a ``reasoning`` step to the prompt.
    The returned prediction carries both ``result.reasoning`` and
    ``result.answer``, which is useful for debugging or auditing model
    decisions.

    turboquant-v2 is enabled here because ChainOfThought generates long
    reasoning chains: the KV cache grows with every output token, so 4-bit
    quantisation can cut KV memory ~4× compared to fp16 without measurable
    quality loss on factual reasoning tasks.

    NOTE: ``kv_cache="turboquant-v2"`` is an alias for the production-stable
    LEAN preset (no QR rotation).  The experimental rotated variant
    (``"turboquant-v2-rotated"``) currently produces garbage output and is
    not recommended.
    """
    print("dspy.ChainOfThought — structured reasoning + answer extraction.")
    print("KV cache: turboquant-v2 (4-bit, LEAN) to reduce memory on long traces.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    question = "What are the three laws of thermodynamics?"

    # --- Pass 1: standard fp16/bf16 KV cache (mlx-lm default) ---
    # cache=False so the second LM's identical prompt actually runs forward()
    # (DSPy's response cache is keyed by model+messages+temp+max_tokens, which
    # would otherwise return the plain answer and skip the turboquant pass).
    lm_plain = AppleLocalLM(_DEFAULT_MODEL, cache=False)
    dspy.configure(lm=lm_plain)
    cot = dspy.ChainOfThought("question -> answer")
    result_plain = cot(question=question)
    bytes_plain = lm_plain.last_kv_bytes  # None — no kv_cache strategy

    # --- Pass 2: turboquant-v2 (4-bit LEAN), public API only ---
    lm_turbo = AppleLocalLM(_DEFAULT_MODEL, kv_cache="turboquant-v2", cache=False)
    dspy.configure(lm=lm_turbo)
    cot = dspy.ChainOfThought("question -> answer")
    result_turbo = cot(question=question)
    bytes_turbo = lm_turbo.last_kv_bytes

    # --- Results ---
    print(f"[CoT] Answer (plain):      {result_plain.answer[:120]}...")
    print(f"[CoT] Answer (turboquant): {result_turbo.answer[:120]}...")
    print()
    if bytes_turbo is not None:
        print(f"[KV] TurboQuant-v2 cache: {bytes_turbo / 1024:.1f} KB (last forward())")
    print(
        "[KV] Plain fp16 KV memory is not directly reported by mlx-lm's default\n"
        "     cache; TurboQuantV2Cache stores K/V as 4-bit packed uint32 with\n"
        "     scale/bias — typically ~4× smaller than fp16 at the same context length."
    )


# ---------------------------------------------------------------------------
# Pattern 7: dspy.ReAct with tools
# ---------------------------------------------------------------------------


def _add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def _lookup_capital(country: str) -> str:
    """Return the capital city of the given country."""
    _data = {
        "france": "Paris",
        "japan": "Tokyo",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
    }
    return _data.get(country.strip().lower(), "Unknown")


def pattern_7_react() -> None:
    """Use dspy.ReAct to answer questions by calling Python tools.

    ReAct interleaves reasoning ("think") and action (tool call) steps until
    it reaches a final answer. The full trajectory — thoughts, tool names,
    tool args, and observations — is available on ``result.trajectory``.
    """
    print("dspy.ReAct — agentic tool use with trajectory inspection.")
    print("The model calls _add and _lookup_capital to answer each question.\n")
    import dspy

    from apple_basefm import AppleLocalLM
    from apple_basefm._hardware import detect_hardware

    model = _DEFAULT_MODEL
    if model == LARGE_MODEL:
        hw = detect_hardware()
        # 20B Q4 weights ≈ 12 GB; with OS + KV cache overhead 24 GB is the safe
        # floor. Below that, Metal is likely to OOM and abort the process.
        _LARGE_MODEL_MIN_RAM_GB = 24
        if hw["is_apple_silicon"] and hw["ram_gb"] < _LARGE_MODEL_MIN_RAM_GB:
            print(
                f"Warning: {hw['ram_gb']} GB unified memory detected. "
                f"The large model ({LARGE_MODEL!r}) needs at least "
                f"{_LARGE_MODEL_MIN_RAM_GB} GB and may trigger a Metal OOM abort."
            )
            try:
                reply = input("Switch to small model instead? [Y/n] ").strip().lower()
            except EOFError:
                reply = "y"  # non-interactive (e.g. piped) — default to safe
            if reply in ("", "y", "yes"):
                model = SMALL_MODEL
                print(f"Using small model: {SMALL_MODEL!r}\n")
            else:
                print("Proceeding with large model — watch for OOM.\n")
        elif not hw["is_apple_silicon"] and model == LARGE_MODEL:
            print(
                f"Warning: not running on Apple Silicon; "
                f"large model ({LARGE_MODEL!r}) may fail.\n"
            )

    lm = AppleLocalLM(model)

    # dspy.context keeps this scoped — global configure is not changed.
    with dspy.context(lm=lm):
        react = dspy.ReAct("question -> answer", tools=[_add, _lookup_capital])

        result = react(question="What is 17 + 25?")
        print("[ReAct] Answer:     ", result.answer)
        print("[ReAct] Steps taken:", len([k for k in result.trajectory if k.startswith("thought_")]))

        result2 = react(question="What is the capital of Japan?")
        print("[ReAct] Answer:     ", result2.answer)
        # Inspect the first trajectory step
        traj = result2.trajectory
        print("[ReAct] Step 0 tool:", traj.get("tool_name_0"))
        print("[ReAct] Step 0 obs: ", traj.get("observation_0"))


# ---------------------------------------------------------------------------
# Pattern 8: Custom dspy.Module pipeline
# ---------------------------------------------------------------------------


def pattern_8_custom_module() -> None:
    """Build a multi-step pipeline by subclassing dspy.Module.

    A ``dspy.Module`` lets you chain sub-predictors (Predict, ChainOfThought,
    ReAct, …) into a single reusable object. Each sub-predictor is declared in
    ``__init__`` and called in ``forward()``.

    This example chains:
      1. An extraction step (raw text → entities)
      2. A reasoning step (entities → verdict)
    """
    print("Custom dspy.Module — two-stage Predict pipeline.")
    print("Stage 1 extracts entities; stage 2 reasons to a verdict.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    class ExtractThenReason(dspy.Module):
        def __init__(self) -> None:
            self.extract = dspy.Predict("raw_text -> entities, dates")
            self.reason = dspy.Predict("entities, dates -> verdict")

        def forward(self, raw_text: str) -> dspy.Prediction:
            extracted = self.extract(raw_text=raw_text)
            verdict = self.reason(entities=extracted.entities, dates=extracted.dates)
            # Attach intermediate fields so callers can inspect them.
            verdict.entities = extracted.entities
            verdict.dates = extracted.dates
            return verdict

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    pipeline = ExtractThenReason()
    result = pipeline(
        raw_text=(
            "Apple announced the M4 chip on May 7, 2024. "
            "The new chip features a 10-core CPU and 10-core GPU."
        )
    )
    print("[Module] Entities:", result.entities)
    print("[Module] Dates:   ", result.dates)
    print("[Module] Verdict: ", result.verdict)


# ---------------------------------------------------------------------------
# Pattern 9: dspy.context — per-call LM scoping
# ---------------------------------------------------------------------------


def pattern_9_context_scoping() -> None:
    """Use dspy.context to override the LM for one block without touching globals.

    ``dspy.configure(lm=...)`` sets a process-wide default; ``dspy.context``
    scopes the override to a ``with`` block and restores the previous LM on
    exit.  Contexts can be nested for fine-grained per-call control.
    """
    print("dspy.context — scoped LM override, no global side effects.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm_a = AppleLocalLM(_DEFAULT_MODEL)
    lm_b = AppleLocalLM("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

    qa = dspy.Predict("question -> answer")

    # Use lm_a for one call …
    with dspy.context(lm=lm_a):
        result_a = qa(question="Name the tallest mountain on Earth.")
        print("[Context/lm_a] Answer:", result_a.answer)

    # … then lm_b for another, without polluting global state.
    with dspy.context(lm=lm_b):
        result_b = qa(question="What is the speed of light in km/s?")
        print("[Context/lm_b] Answer:", result_b.answer)

    print("[Context] Global LM unchanged:", dspy.settings.lm)


# ---------------------------------------------------------------------------
# Pattern 10: AppleFoundationLM + dspy.ReAct (native tool calling)
# ---------------------------------------------------------------------------


def pattern_10_foundation_react() -> None:
    """Use AppleFoundationLM with dspy.ReAct for native on-device tool calling.

    AppleFoundationLM supports ``supports_function_calling=True``: tools are
    converted to native Apple framework Tool subclasses via
    ``_dspy_tool_to_apple_tool()`` and dispatched by the system model —
    no prompt hacking required.

    Requirements: macOS 26+, Apple Intelligence enabled.
    """
    print("AppleFoundationLM + dspy.ReAct — native tool calling (macOS 26+).")
    print("Requires: pip install 'apple-basefm[foundation,apple-fm-sdk,dspy]'\n")
    import platform

    if platform.system() != "Darwin":
        print("[Foundation/ReAct] Skipping: macOS required.")
        return

    try:
        import dspy

        from apple_basefm import AppleFoundationLM

        lm = AppleFoundationLM()

        with dspy.context(lm=lm):
            react = dspy.ReAct("question -> answer", tools=[_add, _lookup_capital])

            def _print_trajectory(traj: dict, label: str) -> None:
                n = len([k for k in traj if k.startswith("thought_")])
                print(f"[{label}] Steps: {n}")
                for i in range(n):
                    thought = traj.get(f"thought_{i}", "")
                    tool = traj.get(f"tool_name_{i}", "")
                    args = traj.get(f"tool_args_{i}", {})
                    obs = traj.get(f"observation_{i}", "")
                    print(f"  step {i}: thought='{thought}'")
                    print(f"           tool={tool}  args={args}  → obs={obs!r}")

            result = react(question="What is 42 + 58?")
            print("[Foundation/ReAct] Answer:", result.answer)
            _print_trajectory(result.trajectory, "Foundation/ReAct")

            print()

            result2 = react(question="What is the capital of Italy?")
            print("[Foundation/ReAct] Answer:", result2.answer)
            _print_trajectory(result2.trajectory, "Foundation/ReAct")
    except (ImportError, RuntimeError) as exc:
        print(f"[Foundation/ReAct] Not available: {exc}")


# ---------------------------------------------------------------------------
# Pattern 11: Streaming via dspy.streamify
# ---------------------------------------------------------------------------


def pattern_11_streaming() -> None:
    """Stream token-by-token output from AppleLocalLM using dspy.streamify.

    ``dspy.streamify(module)`` wraps any DSPy module so that its ``acall()``
    method yields ``StreamResponse`` chunks as they arrive, rather than
    blocking until generation completes. This is useful for building
    interactive UIs or piping output to downstream consumers.

    ``asyncio.run()`` is used here for simplicity; in an already-async context
    use ``await module.acall(...)`` directly.
    """
    print("dspy.streamify — token-by-token streaming output.")
    print("Chunks are printed as they arrive; final answer follows.\n")
    import asyncio

    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)

    async def _run() -> None:
        qa = dspy.Predict("question -> answer")
        streaming_qa = dspy.streamify(qa)

        print("[Stream] Tokens: ", end="", flush=True)
        final = None
        async for chunk in streaming_qa(
            question="List three planets in our solar system.",
            lm=lm,
        ):
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", "") or ""
                print(token, end="", flush=True)
            else:
                # Final StreamResponse — carries the completed Prediction
                final = chunk
        print()  # newline after streamed tokens
        if final is not None and hasattr(final, "answer"):
            print("[Stream] Parsed answer:", final.answer)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Pattern 12: MCP tools with dspy.ReAct
# ---------------------------------------------------------------------------


def pattern_12_mcp_tools() -> None:
    """Use dspy.Tool.from_mcp_tool to wire a FastMCP server into dspy.ReAct.

    ``dspy.Tool.from_mcp_tool(session, tool)`` converts any MCP tool into a
    DSPy-compatible async callable. ReAct dispatches tool calls via the MCP
    session, keeping your tool definitions in one place (the FastMCP server)
    and reusable across any consumer — tests, scripts, or other frameworks.

    This example defines a small in-process FastMCP server with an ``add``
    tool and connects to it via a FastMCP ``Client``. The resulting
    ``dspy.Tool`` objects are passed directly to ``dspy.ReAct``.

    Uses ``react.acall()`` (the async path) because MCP tool calls are
    inherently async (``await session.call_tool(...)`` internally).

    Requirements: pip install "fastmcp>=2.0"
    """
    print("dspy.Tool.from_mcp_tool — MCP tools wired into dspy.ReAct.")
    print("Tools live on a FastMCP server; ReAct calls them via the MCP session.\n")
    import asyncio

    import dspy

    from apple_basefm import AppleLocalLM

    try:
        from fastmcp import FastMCP
        from fastmcp.client import Client
    except ImportError:
        print("[MCP] fastmcp not installed — pip install 'fastmcp>=2.0'")
        return

    # ── Define a minimal FastMCP server ─────────────────────────────────────
    server = FastMCP("example-tools")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    @server.tool
    def lookup_capital(country: str) -> str:
        """Return the capital city of the given country (france/japan/germany)."""
        _data = {"france": "Paris", "japan": "Tokyo", "germany": "Berlin"}
        return _data.get(country.strip().lower(), "Unknown")

    # ── Wire MCP tools into dspy.ReAct and run ───────────────────────────────
    lm = AppleLocalLM(_DEFAULT_MODEL)

    async def _run() -> None:
        async with Client(server) as client:
            tools_raw = await client.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(client.session, t) for t in tools_raw
            ]

            react = dspy.ReAct("question -> answer", tools=dspy_tools)

            with dspy.context(lm=lm):
                result = await react.acall(question="What is 17 + 25?")
                print("[MCP/ReAct] Answer:", result.answer)
                print("[MCP/ReAct] Steps: ", len([k for k in result.trajectory if k.startswith("thought_")]))

                result2 = await react.acall(question="What is the capital of Japan?")
                print("[MCP/ReAct] Answer:", result2.answer)
                print("[MCP/ReAct] Observation:", result2.trajectory.get("observation_0"))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Pattern 13: Class-based dspy.Signature with InputField / OutputField
# ---------------------------------------------------------------------------


def pattern_13_typed_signature() -> None:
    """Define a typed Signature class with InputField/OutputField descriptions.

    Class-based signatures let you attach field descriptions, type annotations,
    and a task-level docstring (the class docstring becomes the LM instruction)
    all in one place.  This is the preferred way to capture domain knowledge
    alongside the signature.
    """
    print("dspy.Signature class — typed fields with descriptions.")
    print("The class docstring becomes the LM's task instruction.\n")
    import dspy
    from typing import Literal

    from apple_basefm import AppleLocalLM

    class SentimentAnalysis(dspy.Signature):
        """Classify the sentiment of a product review and estimate confidence."""

        review: str = dspy.InputField(desc="The product review text to classify.")
        sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(
            desc="Sentiment label: one of positive, negative, or neutral."
        )
        confidence: float = dspy.OutputField(
            desc="Confidence score between 0.0 (uncertain) and 1.0 (certain)."
        )

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    predict = dspy.Predict(SentimentAnalysis)
    result = predict(review="The M4 MacBook Pro is blazing fast and whisper silent.")
    print("[TypedSig] Sentiment:  ", result.sentiment)
    print("[TypedSig] Confidence: ", result.confidence)


# ---------------------------------------------------------------------------
# Pattern 14: dspy.BestOfN — sample N, return highest-reward answer
# ---------------------------------------------------------------------------


def pattern_14_best_of_n() -> None:
    """Sample N candidate answers and return the one with the highest reward.

    BestOfN runs the wrapped module N times at temperature=1.0 and scores
    each prediction with reward_fn(inputs, pred) → float.  The prediction
    with the highest score above threshold is returned.  Use this to improve
    output quality without fine-tuning.
    """
    print("dspy.BestOfN — sample N times, keep best by reward function.")
    print("Runs the wrapped module 3× and picks the longest answer.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    def length_reward(inputs: dict, pred: dspy.Prediction) -> float:
        """Prefer longer, more detailed answers (word count as quality proxy)."""
        return float(len(pred.answer.split()))

    base_module = dspy.ChainOfThought("question -> answer")
    best = dspy.BestOfN(
        module=base_module,
        N=3,
        reward_fn=length_reward,
        threshold=1.0,
    )

    result = best(question="Explain why the sky is blue.")
    print("[BestOfN] Answer:", result.answer[:300])
    print("[BestOfN] (selected best of 3 independent samples)")


# ---------------------------------------------------------------------------
# Pattern 15: dspy.Refine — iterative refinement with per-attempt feedback
# ---------------------------------------------------------------------------


def pattern_15_refine() -> None:
    """Iteratively refine the output until a reward threshold is met.

    Refine runs the module up to N times, feeding failure feedback back into
    each subsequent attempt, stopping early when reward_fn returns a value
    above threshold.  Use this when the first answer is often correct but
    sometimes needs nudging toward a stricter quality criterion.
    """
    print("dspy.Refine — retry with feedback until reward threshold is met.")
    print("Stops early once the answer contains causal language.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    def causal_reward(inputs: dict, pred: dspy.Prediction) -> float:
        """Reward 1.0 if the answer includes causal language; else 0.0."""
        lower = pred.answer.lower()
        return 1.0 if any(w in lower for w in ("because", "due to", "since", "caused by")) else 0.0

    base_module = dspy.ChainOfThought("question -> answer")
    refine = dspy.Refine(
        module=base_module,
        N=3,
        reward_fn=causal_reward,
        threshold=0.9,
    )

    result = refine(question="Why do apples fall from trees?")
    print("[Refine] Answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 16: dspy.MultiChainComparison — M CoT chains, holistic comparison
# ---------------------------------------------------------------------------


def pattern_16_multi_chain_comparison() -> None:
    """Run M CoT chains independently, then pick the most consistent answer.

    MultiChainComparison generates M separate chain-of-thought completions
    and passes them to a comparison predictor that produces the most accurate
    holistic reasoning.  This reduces variance compared to a single CoT pass.
    """
    print("dspy.MultiChainComparison — compare M independent CoT chains.")
    print("Three chains are generated; the comparator picks the best reasoning.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    M = 3
    question = "What is the boiling point of water at sea level in Celsius?"

    cot = dspy.ChainOfThought("question -> answer")
    completions = [cot(question=question) for _ in range(M)]

    comparator = dspy.MultiChainComparison("question -> answer", M=M)
    result = comparator(completions, question=question)
    rationale = getattr(result, "rationale", None) or ""
    print("[MultiChain] Rationale:", rationale[:150])
    print("[MultiChain] Answer:   ", result.answer)


# ---------------------------------------------------------------------------
# Pattern 17: dspy.ProgramOfThought — generate + execute Python code
# ---------------------------------------------------------------------------


def pattern_17_program_of_thought() -> None:
    """Instruct the LM to write Python code that computes the answer, then run it.

    ProgramOfThought is best for arithmetic, unit conversions, or any task
    where executing code is more reliable than prose reasoning.  The generated
    code runs in a sandboxed local Python interpreter (Deno/Pyodide).

    Requires: Deno  https://docs.deno.com/runtime/getting_started/installation/
      brew install deno   # macOS
    """
    import shutil

    if shutil.which("deno") is None:
        print("dspy.ProgramOfThought — SKIPPED (Deno not installed).")
        print("Install with: brew install deno")
        print("See: https://docs.deno.com/runtime/getting_started/installation/")
        return

    print("dspy.ProgramOfThought — LM writes Python; code runs locally.")
    print("Best for arithmetic and data tasks where code beats prose.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    pot = dspy.ProgramOfThought("question -> answer")
    result = pot(question="What is 15 * 7 + 33?")
    print("[PoT] Answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 18: dspy.Parallel — concurrent module calls across a thread pool
# ---------------------------------------------------------------------------


def pattern_18_parallel() -> None:
    """Run multiple DSPy module calls concurrently across a thread pool.

    dspy.Parallel distributes a list of (module, inputs-dict) pairs across
    num_threads worker threads and returns results in input order.  Use this
    to maximise throughput for batch labelling, parallel re-ranking, or any
    scenario with many independent calls.
    """
    print("dspy.Parallel — concurrent (module, inputs) pairs across threads.")
    print("Three questions answered in parallel; results returned in order.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")
    questions = [
        "What is the capital of France?",
        "What is 12 * 12?",
        "Name one planet in our solar system.",
    ]

    parallel = dspy.Parallel(num_threads=3, disable_progress_bar=True)
    exec_pairs = [(qa, {"question": q}) for q in questions]
    results = parallel(exec_pairs)

    for q, r in zip(questions, results):
        print(f"[Parallel] Q: {q!r}")
        print(f"           A: {r.answer}")


# ---------------------------------------------------------------------------
# Pattern 19: dspy.CodeAct — code-based agentic tool use
# ---------------------------------------------------------------------------


def pattern_19_code_act() -> None:
    """Write and execute Python code in each agent step to call tools.

    Unlike ReAct (JSON-arg dispatch), CodeAct has the LM emit a Python code
    snippet at every step.  The snippet runs in a sandboxed local interpreter;
    tool functions are injected into the interpreter's namespace so the model
    can call them directly.  Code runs locally — review generated snippets
    before deploying in production.
    """
    print("dspy.CodeAct — LM emits Python code snippets; tools run locally.")
    print("Tools (_add, _lookup_capital) are injected into the interpreter namespace.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    # _add and _lookup_capital are module-level functions defined above;
    # CodeAct retrieves their source via inspect.getsource.
    code_act = dspy.CodeAct(
        "question -> answer",
        tools=[_add, _lookup_capital],
    )

    result = code_act(question="What is 23 + 19?")
    print("[CodeAct] Answer:", result.answer)

    result2 = code_act(question="What is the capital of Germany?")
    print("[CodeAct] Answer:", result2.answer)


# ---------------------------------------------------------------------------
# Pattern 20: dspy.RLM — REPL-history reinforcement-learning loop
# ---------------------------------------------------------------------------


def pattern_20_rlm() -> None:
    """Run a reinforcement-learning style code-interpreter agent loop.

    RLM iteratively generates Python code, executes it in a persistent REPL,
    observes the output, and refines until the signature's outputs are filled.
    Unlike ProgramOfThought (single block) or CodeAct (ReAct-style), RLM
    maintains a full REPL history across many small code calls.  Code runs
    locally — review generated snippets before deploying in production.
    """
    print("dspy.RLM — REPL-history code interpreter, reinforcement-learning loop.")
    print("Iteratively generates + executes Python, accumulating a REPL history.\n")
    import dspy

    from apple_basefm import AppleLocalLM

    lm = AppleLocalLM(_DEFAULT_MODEL)
    dspy.configure(lm=lm)

    rlm = dspy.RLM(
        "question -> answer",
        tools=[_add, _lookup_capital],
        max_iterations=10,
        max_llm_calls=20,
    )

    result = rlm(question="What is 7 + 8?")
    print("[RLM] Answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 21: Raw mlx-lm forward pass — access logits without AppleLocalLM
# ---------------------------------------------------------------------------
# Use this pattern when you need access to raw logits tensors: embedding
# extraction, custom sampling strategies, probability analysis, or any task
# that does not fit the chat-completion interface provided by AppleLocalLM.


def pattern_21_raw_mlx_forward() -> None:
    """Load a model with mlx_lm and perform a raw forward pass for logits.

    This bypasses AppleLocalLM entirely, giving direct access to the logits
    tensor.  Useful for:
      * Embedding extraction (hidden states at any layer)
      * Custom sampling or constrained decoding
      * Token probability analysis
      * Any task that doesn't fit the chat-completion interface

    Also demonstrates mlx_lm.generate() as a convenience wrapper for text
    output after raw loading.
    """
    print("mlx_lm.load() + model() — raw logits without AppleLocalLM.")
    print("Use this for embeddings, custom decoding, or low-level analysis.\n")
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        print("[RawMLX] mlx / mlx-lm not installed — pip install 'apple-basefm[mlx]'")
        return

    model, tokenizer = mlx_lm.load(_DEFAULT_MODEL)

    # ── Tokenize ─────────────────────────────────────────────────────────────
    prompt = "The capital of France is"
    token_ids = tokenizer.encode(prompt)
    input_ids = mx.array([token_ids])           # shape: (1, seq_len)

    # ── Raw forward pass → logits ─────────────────────────────────────────────
    logits = model(input_ids)                   # shape: (1, seq_len, vocab_size)
    mx.eval(logits)                             # materialise the lazy graph

    next_token_logits = logits[0, -1, :]        # logits for the next token position
    next_token_id = mx.argmax(next_token_logits).item()
    print(f"[RawMLX] Greedy next-token id: {next_token_id}")

    # ── Convenience text generation ──────────────────────────────────────────
    messages = [{"role": "user", "content": "Name three planets."}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text = mlx_lm.generate(model, tokenizer, prompt=formatted, max_tokens=64)
    print("[RawMLX] Generated:", text)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="apple-basefm usage examples")
    parser.add_argument(
        "--pattern",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                 "13", "14", "15", "16", "17", "18", "19", "20", "21", "all"],
        default="all",
        help="Which pattern to run (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["small", "large"],
        default="small",
        help=f"Local model size: small={SMALL_MODEL!r}, large={LARGE_MODEL!r} (default: small)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the DSPy response cache (disk + memory) before running examples.",
    )
    args = parser.parse_args()

    if args.clear_cache:
        import dspy as _dspy
        try:
            _dspy.cache.reset_memory_cache()
            if _dspy.cache.enable_disk_cache and hasattr(_dspy.cache.disk_cache, "clear"):
                _dspy.cache.disk_cache.clear()
            print(f"DSPy cache cleared ({_dspy.cache.disk_cache_dir})\n")
        except Exception as _e:
            print(f"Warning: could not clear DSPy cache: {_e}\n")

    _DEFAULT_MODEL = LARGE_MODEL if args.model == "large" else SMALL_MODEL

    if args.model == "large" and args.pattern == "all":
        print(
            "Warning: running all patterns with a large model requires substantial\n"
            "unified memory. If the process aborts with a Metal OOM error, run\n"
            "individual patterns (--pattern N) or switch to --model small.\n"
        )

    runners = {
        "1": pattern_1_standalone,
        "2": pattern_2_full_dspy,
        "3": pattern_3_mixed_pipeline,
        "4": pattern_4_apple_intelligence,
        "5": pattern_6_chain_of_thought,
        "6": pattern_7_react,
        "7": pattern_8_custom_module,
        "8": pattern_9_context_scoping,
        "9": pattern_10_foundation_react,
        "10": pattern_11_streaming,
        "11": pattern_12_mcp_tools,
        "12": pattern_13_typed_signature,
        "13": pattern_14_best_of_n,
        "14": pattern_15_refine,
        "15": pattern_16_multi_chain_comparison,
        "16": pattern_17_program_of_thought,
        "17": pattern_18_parallel,
        "18": pattern_19_code_act,
        "19": pattern_20_rlm,
        "20": pattern_21_raw_mlx_forward,
        "21": pattern_5_token_session,
    }

    _FOUNDATION_PATTERNS = {"4", "9"}

    if args.pattern == "all":
        items = list(runners.items())
        for i, (name, fn) in enumerate(items):
            model_label = "AppleFoundationLM (system)" if name in _FOUNDATION_PATTERNS else _DEFAULT_MODEL
            print(f"\n{'=' * 60}")
            print(f"Pattern {name}  |  model: {model_label}")
            print("=" * 60)
            fn()
            if i < len(items) - 1:
                try:
                    input("\nPress Enter to continue to the next pattern…")
                except EOFError:
                    pass  # non-interactive (pipe / CI) — continue silently
    else:
        runners[args.pattern]()
