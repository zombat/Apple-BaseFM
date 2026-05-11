"""dspy-apple: Apple Silicon and Apple Intelligence backends for DSPy.

Provides:
  AppleFoundationLM  — macOS 26+, Apple Intelligence, native @generable + fm.Tool
  AppleLocalLM       — mlx-lm, any HuggingFace model on Apple Silicon
"""

try:
    from apple_basefm.apple_fm import AppleFoundationLM
except (ImportError, RuntimeError):
    AppleFoundationLM = None  # type: ignore[assignment, misc]

try:
    from apple_basefm.apple_local import AppleLocalLM
except (ImportError, RuntimeError, NotImplementedError):
    AppleLocalLM = None  # type: ignore[assignment, misc]

from apple_basefm._compat import DSPY_AVAILABLE

__version__ = "0.1.0"
__all__ = ["AppleFoundationLM", "AppleLocalLM", "DSPY_AVAILABLE", "__version__"]
