"""dspy-apple: Apple Silicon and Apple Intelligence backends for DSPy.

Provides:
  AppleFoundationLM  — macOS 26+, Apple Intelligence, native @generable + fm.Tool
  AppleLocalLM       — mlx-lm, any HuggingFace model on Apple Silicon
  list_mlx_models    — list locally cached MLX-compatible models
  suggest_models     — suggest models suited to local Apple Silicon hardware
  detect_hardware    — detect chip, RAM, and free disk on the local Mac
"""

try:
    from apple_basefm.apple_fm import AppleFoundationLM
except (ImportError, RuntimeError):
    AppleFoundationLM = None  # type: ignore[assignment, misc]

try:
    from apple_basefm.apple_local import AppleLocalLM
except (ImportError, RuntimeError, NotImplementedError):
    AppleLocalLM = None  # type: ignore[assignment, misc]

from apple_basefm._catalog import ModelEntry, SuggestResult, suggest_models
from apple_basefm._cli import MLXModelInfo, list_mlx_models
from apple_basefm._compat import DSPY_AVAILABLE
from apple_basefm._hardware import HardwareInfo, detect_hardware
from apple_basefm._logging import OSLogHandler, configure_logging
from apple_basefm._session import _SessionAccumulator, token_session

__version__ = "0.3.0"
__all__ = [
    "AppleFoundationLM",
    "AppleLocalLM",
    "DSPY_AVAILABLE",
    "HardwareInfo",
    "MLXModelInfo",
    "ModelEntry",
    "OSLogHandler",
    "SuggestResult",
    "configure_logging",
    "detect_hardware",
    "list_mlx_models",
    "suggest_models",
    "token_session",
    "_SessionAccumulator",
    "__version__",
]
