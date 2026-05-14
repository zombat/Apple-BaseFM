"""Safe, simple MCP tools for use in apple-basefm tests.

All tools are pure-Python with no I/O, no eval, and no side effects.
They are registered on a FastMCP server instance (``MCP_SERVER``) so tests
can connect to it in-process via ``fastmcp.client.Client(MCP_SERVER)``
without spawning a subprocess or opening a network socket.

The raw callables are also importable directly for use as DSPy ReAct tools::

    from tests.tools import add, multiply, lookup_capital
    react = dspy.ReAct("question -> answer", tools=[add, multiply])
"""
from __future__ import annotations

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

MCP_SERVER = FastMCP("apple-basefm-test")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@MCP_SERVER.tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@MCP_SERVER.tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


@MCP_SERVER.tool
def lookup_capital(country: str) -> str:
    """Return the capital city of the given country (lower-cased lookup).

    Supported countries: france, japan, germany, italy, spain.
    Returns ``"Unknown"`` for any unrecognised input.
    """
    _CAPITALS = {
        "france": "Paris",
        "japan": "Tokyo",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
    }
    return _CAPITALS.get(country.strip().lower(), "Unknown")


@MCP_SERVER.tool
def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


@MCP_SERVER.tool
def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))
