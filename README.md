# Apple-BaseFM

Apple Silicon and Apple Intelligence language model backends for [DSPy](https://github.com/stanfordnlp/dspy).

Extracted from [DSPy PR #9473](https://github.com/stanfordnlp/dspy/pull/9473) into a standalone PyPI package.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/raymondandrewrizzo)

---

## What's included

| Class | Backend | Platform |
|---|---|---|
| `AppleFoundationLM` | Apple Intelligence system model | macOS 26+ with Apple Intelligence |
| `AppleLocalLM` | Any mlx-lm model (HF repo or local dir) | macOS 14+ on Apple Silicon |

Both classes are fully-conformant `dspy.BaseLM` subclasses when DSPy is installed,
or usable standalone with a minimal stub when it is not.

---

## Installation

### Minimal (standalone, no DSPy)

```bash
pip install apple-basefm
```

### With DSPy

```bash
pip install "apple-basefm[dspy]"
```

### MLX backend (local models)

```bash
pip install "apple-basefm[mlx,dspy]"
```

### Apple Foundation Models (`AppleFoundationLM`)

Install on a Mac running macOS 26+ with Apple Intelligence enabled.
Setup guide: https://apple.github.io/python-apple-fm-sdk/getting_started.html

```bash
pip install "apple-basefm[foundation,apple-fm-sdk,dspy]"
```

---

## Quick starts

### 1. Standalone — no DSPy required

```python
from apple_basefm import AppleLocalLM

lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
response = lm.forward(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)
```

### 2. Full DSPy integration

```python
import dspy
from apple_basefm import AppleLocalLM

lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
print(qa(question="Explain quantum entanglement in one sentence.").answer)
```

### 3. Mixed pipeline — local preprocessing + cloud reasoning

```python
import dspy
from apple_basefm import AppleLocalLM

local_lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
cloud_lm = dspy.LM("openai/gpt-4o-mini")

class ExtractThenReason(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("raw_text -> entities, dates", lm=local_lm)
        self.reason  = dspy.Predict("entities, dates -> verdict",  lm=cloud_lm)

    def forward(self, raw_text):
        extracted = self.extract(raw_text=raw_text)
        return self.reason(entities=extracted.entities, dates=extracted.dates)

pipeline = ExtractThenReason()
result = pipeline.forward(raw_text="Apple announced the M4 chip on May 7, 2024.")
print(result.verdict)
```

---

## AppleFoundationLM

Requires macOS 26+ with Apple Intelligence and the apple-fm-sdk.

```python
import dspy
from apple_basefm import AppleFoundationLM

lm = AppleFoundationLM()
dspy.configure(lm=lm)

from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

qa = dspy.Predict("text -> sentiment_label, confidence_score")
result = qa(text="I absolutely love Apple Silicon!")
print(result.sentiment_label, result.confidence_score)
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | `"apple/on-device"` | Identifier stored in cache keys / history |
| `temperature` | `None` | Passed to `GenerationOptions`; `None` uses model default |
| `max_tokens` | `None` | Reserved; stored but not yet wired in SDK |
| `cache` | `True` | Enable DSPy request cache |
| `timeout` | `120.0` | Max seconds per `session.respond()` call; `None` disables |

---

## AppleLocalLM

```python
from apple_basefm import AppleLocalLM

lm = AppleLocalLM(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    temperature=0.0,
    max_tokens=1024,
    max_concurrency=1,  # sequential is safe; >1 requires thread-safe model
)
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | _(required)_ | HuggingFace repo ID or absolute path to local MLX dir |
| `backend` | `"mlx"` | Only `"mlx"` is implemented; `"coreml"` raises `NotImplementedError` |
| `bits` | `None` | Informational quantization hint; does not trigger quantization |
| `temperature` | `0.0` | Sampling temperature; clamped to `[0.0, 2.0]` |
| `max_tokens` | `1000` | Max tokens per call; floored at `1` |
| `cache` | `True` | Enable DSPy request cache |
| `max_concurrency` | `1` | Semaphore limit for concurrent `aforward()` calls |
| `kv_cache` | `None` | KV cache strategy: `"turboquant-v2"`, `"turboquant-v2-lean"`, a `KVCacheStrategy` instance, or `None` |

---

## TurboQuant V2 KV Cache

TurboQuant V2 is an optional KV cache backend for `AppleLocalLM` that compresses the
attention key/value cache to 4-bit (or 2/8-bit), achieving ~3.6× memory reduction at
4-bit. It is invisible to DSPy — the same optimizer and module code works unchanged.

### When it helps

KV cache compression is most impactful in the **64–128 GB** memory tier, where a 70B
model fills most of unified memory and optimizer loops inject long few-shot prompts that
push context to 4–8K tokens. Without compression the cache grows linearly and starts
competing with model weights for memory bandwidth. With TurboQuant V2 that pressure is
3.6× smaller, keeping generation speed flat as context grows.

At 8–16 GB the bottleneck is model weights, not KV cache — TurboQuant provides no
practical benefit there.

### Speed impact

Near zero. `mx.quantized_matmul` replaces `mx.matmul` at essentially the same cost and
the quantize/dequantize step is not on the generation critical path. TurboQuant V2 4-bit
runs at ~105% of fp16 baseline in generation throughput. The real comparison is not
_TurboQuant vs. no TurboQuant on speed_ — it is what happens to speed as context length
grows. TurboQuant keeps the curve flat.

### Usage

```python
from apple_basefm import AppleLocalLM

# Preset strings (recommended)
lm = AppleLocalLM(
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    kv_cache="turboquant-v2",      # 4-bit, no rotation until attention_v2 ships
)

# LEAN mode — no rotation, numerically identical to mlx-lm --kv-bits 4
lm = AppleLocalLM(
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    kv_cache="turboquant-v2-lean",
)

# Custom configuration
from apple_basefm._kv import TurboQuantV2Cache

lm = AppleLocalLM(
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    kv_cache=TurboQuantV2Cache(bits=4, group_size=64, use_rotation=True),
)
```

| Preset | Bits | Rotation | Notes |
|---|---|---|---|
| `"turboquant-v2"` | 4 | No (pending) | Recommended default; rotation will be enabled automatically when `attention_v2` ships |
| `"turboquant-v2-lean"` | 4 | No | Permanent stable alias; always `use_rotation=False`, numerically identical to `mlx-lm --kv-bits 4` |

`TurboQuantV2Cache` valid values: `bits` ∈ `{2, 4, 8}`, `group_size` ≥ 1, `step` ≥ 1.

---

## Apple Silicon Memory Guide

### Suggested models by unified memory

All models are 4-bit quantized (Q4) via `mlx-community` unless noted.
Practical rule: reserve ~4–6 GB for macOS and background processes.

| RAM | Chip examples | Fits without TurboQuant | Fits with TurboQuant V2 4-bit | Notes |
|---|---|---|---|---|
| 8 GB | M1–M5 (base) | Llama 3.2 1B, Phi-3.5 Mini 3.8B, Qwen2.5 3B | Same — KV cache is not the bottleneck here | Marginal for DSPy optimizer loops; keep context short |
| 16 GB | M1–M5 (base/Air) | Llama 3.1 8B, Mistral 7B, Gemma 3 12B, Qwen2.5 7B | Same models at longer context windows | Minimum useful tier for DSPy; optimizer runs are slow |
| 24 GB | M1 Pro, M2 Pro, M3 Pro, M4 iMac, M5 Pro (low) | Llama 3.1 8B, Phi-4 14B, Gemma 3 12B | Qwen3 30B A3B MoE (normally tight at ~16.5 GB; TurboQuant creates breathing room for longer contexts) | Sweet spot for general DSPy use |
| 32 GB | M1/M2 Max, M3 Max (low), M4 Mac Mini (high), M5 (high) | Phi-4 14B, Qwen2.5 14B, Llama 3.1 8B (long ctx) | Gemma 3 27B, Qwen2.5 32B at moderate context | Comfortable for most DSPy optimizer workloads |
| 36 GB | M1 Max, M3 Pro (high), M4 Max (low), M5 Max (low) | Qwen2.5 32B, Gemma 3 27B | Same models at longer context windows | Solid research tier |
| 48 GB | M1 Ultra (low), M2 Max (high), M3 Max (mid), M4 Pro (high), M5 Pro (high) | Qwen2.5 32B, Llama 3.3 70B (tight) | Llama 3.3 70B at useful context lengths; Qwen2.5 32B at very long context | First tier where 70B becomes practical |
| 64 GB | M1 Ultra, M2 Max (high), M3/M4 Max (low), M5 Pro (high), M5 Max (low) | Llama 3.3 70B, Qwen2.5 72B, Mistral Large | Same at significantly longer contexts | Strong optimizer tier; MIPROv2 on 70B viable |
| 96 GB | M2 Ultra (low), M3 Ultra (low), M4 Max (high), M5 Max (mid) | Llama 3.1 70B, Qwen2.5 72B, Gemma 3 27B | 70B at very long context (8K+); mixed pipelines with two models loaded | Multi-model pipelines become practical |
| 128 GB | M2 Ultra, M3 Max (high), M4/M5 Max (high) | Llama 3.1 70B (full ctx), Qwen2.5 72B, first tier for 100B+ | Near-lossless quality at 8K+ on 70B; comfortable headroom for optimizer parallelism | Primary target for serious `AppleLocalLM` use |
| 192 GB | M2 Ultra (high), M3 Ultra (low) | Llama 3.1 70B, Mixtral 8×22B (141B MoE), DeepSeek R1 distill 70B | All above at maximum context; two 70B models simultaneously | Research / production deployment tier |
| 256 GB | M3 Ultra (high), future M5 Ultra | Llama 3.1 405B (quantized), DeepSeek V3, large MoE models | Effectively no KV cache constraint at normal context lengths | Supply constrained as of May 2026 |
| 512 GB | M3 Ultra (max, delisted) | DeepSeek R1 671B (quantized), Llama 3.1 405B full precision | Effectively unlimited for any currently available open model | No longer purchasable new |

#### A few practical notes

**Where TurboQuant matters most** is the 64–128 GB range. At those tiers you can load a
70B model, but optimizer loops inject few-shot examples into every prompt — KV cache grows
fast and starts competing with model weights for the same unified memory pool. TurboQuant
V2 4-bit gives you 3.6× compression on that cache, which directly translates to more
optimizer candidates running in parallel before hitting the `max_concurrency` ceiling.

**MoE caveat:** Models like Qwen3 30B A3B and Mixtral are MoE architectures. They have
large total parameter counts but only activate a fraction per token, so their effective
memory footprint is smaller than the parameter count implies. They punch above their
weight on Apple Silicon specifically because unified memory handles sparse access patterns
well.

**The 8–16 GB floor:** These tiers can run `AppleLocalLM` but are better suited to
`AppleFoundationLM`. The on-device Apple Intelligence model requires no memory budget
from you and is always available on macOS 26+ regardless of installed RAM.

---

### Stack comparison: mlx-lm vs. mlx-lm + TurboQuant V2

#### Generation speed (tokens/sec, representative hardware)

| Stack | M4 Pro 48 GB · 8B model | M4 Pro 48 GB · 32B MoE | M4 Max 128 GB · 70B model | Notes |
|---|---|---|---|---|
| mlx-lm (raw, no TurboQuant) | ~160 tok/s | ~160 tok/s | ~55 tok/s | Direct MLX, no server layer |
| mlx-lm + TurboQuant V2 4-bit | ~155 tok/s (short ctx) / ~155 tok/s (long ctx) | ~155 tok/s (short ctx) / ~155 tok/s (long ctx) | ~54 tok/s (short ctx) / ~54 tok/s (long ctx) | Speed identical to plain mlx-lm at short context; benefit is memory, not speed |

Speed is effectively unchanged at short context. The advantage is that speed stays flat as
context grows — without TurboQuant, KV cache growth competes with model weights for
memory bandwidth and throughput degrades. With TurboQuant that pressure is 3.6× smaller.

#### Memory overhead

| Stack | 8B model footprint | 32B MoE footprint | Notes |
|---|---|---|---|
| mlx-lm (raw) | ~5.0 GB | ~19.5 GB | ~10% lower than GGUF-based tools due to native unified memory |
| mlx-lm + TurboQuant V2 4-bit | ~5.0 GB (weights) + compressed KV | ~19.5 GB (weights) + compressed KV | KV cache at T=8192: 969 MB → 266 MB (3.6×) |

#### When each approach wins

| Scenario | Best choice | Why |
|---|---|---|
| Maximum raw throughput on Mac | mlx-lm (raw) | Lowest overhead |
| DSPy optimizer loops (short prompts, many calls) | mlx-lm (raw) | Lowest per-call latency |
| DSPy optimizer loops (long prompts, 4K+ context) | mlx-lm + TurboQuant V2 | KV cache compression prevents memory pressure from degrading speed as context grows |
| Two models loaded simultaneously | mlx-lm + TurboQuant V2 | 3.6× KV compression frees headroom for the second model's weights |
| MoE models (Qwen3, Mixtral) | mlx-lm (raw or TurboQuant) | MLX handles MoE routing ~2–3× faster than llama.cpp |

---

## CLI: download

Download an MLX model from HuggingFace Hub into the local cache.

```
apple-basefm download REPO_ID [--revision REV] [--dry-run] [--yes]
```

| Argument | Default | Description |
|---|---|---|
| `REPO_ID` | _(required)_ | HuggingFace repo ID, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| `--revision` | `main` | Commit hash, tag, or branch to pin. Use a commit hash for DSPy reproducibility. |
| `--dry-run` | off | Print repo ID and estimated size; do not download. |
| `--yes` | off | Skip the disk-space confirmation prompt (same as `remove`). |

The command prints the final local cache path on success.

### Typical workflow

```bash
# 1. Find a model
apple-basefm suggest

# 2. Download it (paste the REPO ID from suggest output)
apple-basefm download mlx-community/Llama-3.2-3B-Instruct-4bit

# 3. Pin a specific revision for reproducibility
apple-basefm download mlx-community/Llama-3.2-3B-Instruct-4bit --revision a1b2c3d

# 4. Check disk impact before committing
apple-basefm download mlx-community/Llama-3.3-70B-Instruct-4bit --dry-run
```

### Implementation notes (for contributors)

- **Download**: uses `huggingface_hub.snapshot_download(repo_id, revision=..., resume_download=True)`. The hub library handles resumable downloads and emits built-in tqdm progress — no extra dependency needed.
- **Preflight checks** (run before download starts):
  1. `repo_info(repo_id)` — confirms the repo exists on the Hub; surfaces a clear error for typos or private repos.
  2. Disk space — compares estimated model size against `_hardware.detect_hardware().free_disk_gb`. The estimated size comes from the offline catalog `disk_gb` value when the repo ID matches a catalog entry; otherwise from Hub metadata. For `gpt-oss-20b` variants, always use the catalog value (`11.0 GB`) — Hub metadata varies by revision and can be misleading.
- **Input**: strict HuggingFace repo IDs only (v1). Short name aliases (`llama-3.2-3b`) are deferred; they add discoverability but create a maintenance burden when model names change.
- **local_dir_use_symlinks**: pass `local_dir_use_symlinks=False` when using `hf_hub_download` to avoid symlink issues on some filesystems. `snapshot_download` handles this internally.

---

## Development

```bash
git clone https://github.com/zombat/Apple-BaseFM
cd apple-basefm
pip install -e ".[dev]"
pytest tests/ -v          # unit tests (no Apple hardware required)
pytest tests/integration/ # integration tests (requires macOS 26+ / Apple Intelligence)
ruff check apple_basefm/
mypy apple_basefm/
```

---

## Using a HuggingFace Mirror

If `huggingface.co` is blocked or slow in your network, you can point every
`huggingface_hub` call — including model downloads, `apple-basefm mlx-models`,
`apple-basefm suggest`, and `apple-basefm remove` — at a mirror endpoint.

### Setting the mirror endpoint

Set the `HF_ENDPOINT` environment variable before running any command or importing the
library.  The value must be the base URL of the mirror with no trailing slash.

```bash
# Shell (Linux / macOS)
export HF_ENDPOINT=https://hf-mirror.com

# Single command
HF_ENDPOINT=https://hf-mirror.com apple-basefm suggest

# Or in Python before any import
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import apple_basefm
```

`huggingface_hub` reads `HF_ENDPOINT` at import time, so the variable must be set before
the library is first imported in the current process.

### Downloading models through a mirror

Once `HF_ENDPOINT` is set, `mlx-lm` will route all downloads through the mirror because
it uses `huggingface_hub` internally:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from apple_basefm import AppleLocalLM
lm = AppleLocalLM('mlx-community/Llama-3.2-3B-Instruct-4bit')
"
```

Or with `huggingface-cli` directly:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit
```

### CLI commands

All three `apple-basefm` subcommands work unmodified once `HF_ENDPOINT` is set:

```bash
export HF_ENDPOINT=https://hf-mirror.com

apple-basefm mlx-models            # lists locally cached models (no network call)
apple-basefm suggest               # queries mlx-community via the mirror endpoint
apple-basefm suggest --offline     # skips the network call entirely; no mirror needed
apple-basefm remove <repo_id>      # removes from local cache; no network call
```

`suggest` makes a live query to `mlx-community` on HuggingFace Hub (or the mirror). If
the mirror is unavailable, it automatically falls back to the built-in offline catalog.
Use `--offline` to force the offline catalog and skip the network call entirely.

### Persisting the setting

Add the export to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.) or to a `.env` file
loaded by your project:

```bash
# ~/.zshrc
export HF_ENDPOINT=https://hf-mirror.com
```

Or pin it in a `pyproject.toml`-adjacent `.env` that your runner loads:

```ini
HF_ENDPOINT=https://hf-mirror.com
```

### Authentication on private or enterprise mirrors

If the mirror requires a token, use `HF_TOKEN` (same variable `huggingface_hub` uses for
`huggingface.co`):

```bash
export HF_ENDPOINT=https://my-internal-mirror.example.com
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> **Security note**: `HF_TOKEN` is read by `huggingface_hub` and never logged or stored
> by `apple-basefm`. Do not hardcode tokens in source files.

---

## Compatibility matrix

| apple-basefm | DSPy | Python | macOS (local models) | macOS (Foundation) |
|---|---|---|---|---|
| 0.2.x | ≥ 2.5.0 | ≥ 3.11 | 14+ (Apple Silicon) | 26+ (Apple Intelligence) |
| 0.1.x | ≥ 2.5.0 | ≥ 3.11 | 14+ (Apple Silicon) | 26+ (Apple Intelligence) |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Legal

This project contains code derived from [DSPy](https://github.com/stanfordnlp/dspy)
(PR #9473), copyright © 2023 Stanford Future Data Systems, used under the MIT License.

Apple, Apple Intelligence, Apple Silicon, and Foundation Models are trademarks of
Apple Inc. The `apple_fm_sdk` is proprietary Apple software, not included here, and
must be obtained through Apple's developer channels subject to Apple's terms.

[`mlx`](https://github.com/ml-explore/mlx) and
[`mlx-lm`](https://github.com/ml-explore/mlx-lm) are optional dependencies maintained
by Apple Inc., used under the MIT License, and not bundled with this package.

This project is independent and is not affiliated with, endorsed by, or sponsored
by Apple Inc. or Stanford University.
