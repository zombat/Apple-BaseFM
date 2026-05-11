# dspy-apple

Apple Silicon and Apple Intelligence language model backends for [DSPy](https://github.com/stanfordnlp/dspy).

Extracted from [DSPy PR #9473](https://github.com/stanfordnlp/dspy/pull/9473) into a standalone PyPI package.

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
pip install dspy-apple
```

### With DSPy

```bash
pip install "dspy-apple[dspy]"
```

### MLX backend (local models)

```bash
pip install "dspy-apple[mlx,dspy]"
```

### Apple Foundation Models (`AppleFoundationLM`)

The Apple Foundation Models SDK is **not on PyPI**.
Install it from [Apple's developer distribution channel](https://developer.apple.com/documentation/foundationmodels)
on a Mac running macOS 26+.

```bash
pip install "dspy-apple[foundation,dspy]"
# then install apple-fm-sdk separately from Apple
```

---

## Quick starts

### 1. Standalone — no DSPy required

```python
from dspy_apple import AppleLocalLM

lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
response = lm.forward(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)
```

### 2. Full DSPy integration

```python
import dspy
from dspy_apple import AppleLocalLM

lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
print(qa(question="Explain quantum entanglement in one sentence.").answer)
```

### 3. Mixed pipeline — local preprocessing + cloud reasoning

```python
import dspy
from dspy_apple import AppleLocalLM

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
from dspy_apple import AppleFoundationLM

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
from dspy_apple import AppleLocalLM

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

---

## Development

```bash
git clone https://github.com/zombat/Apple-BaseFM
cd dspy-apple
pip install -e ".[dev]"
pytest tests/ -v          # unit tests (no Apple hardware required)
pytest tests/integration/ # integration tests (requires macOS 26+ / Apple Intelligence)
ruff check dspy_apple/
mypy dspy_apple/
```

---

## Compatibility matrix

| dspy-apple | DSPy | Python | macOS (local models) | macOS (Foundation) |
|---|---|---|---|---|
| 0.1.x | ≥ 2.5.0 | ≥ 3.11 | 14+ (Apple Silicon) | 26+ (Apple Intelligence) |

---

## License

MIT — see [LICENSE](LICENSE).
