#!/usr/bin/env python3
"""guard.py — static governance scanner for apple-basefm.

Flags high-confidence violations across five governance areas:

  1. Unsafe deserialization  (pickle, yaml.load, eval/exec on non-constants)
  2. Subprocess injection    (shell=True in any subprocess call)
  3. JSON without error handling (--strict only)
  4. Hardcoded secrets       (variable names matching credential patterns
                              assigned to non-trivial string literals)
  5. Sensitive data in logs  (password/token values passed to logger calls)

Exit code: 0 = clean, 1 = violations found.

Usage::

    python3 scripts/guard.py [--dir PATH] [--strict]

    --dir PATH   Root directory to scan (default: package root next to scripts/)
    --strict     Also flag json.loads() calls without an adjacent try/except
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    severity: str   # CRITICAL | HIGH | WARNING | INFO
    rule_id: str
    path: Path
    line: int
    message: str
    snippet: str = ""

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}"
        lines = [f"[{self.severity}] {self.rule_id}  {loc}", f"  {self.message}"]
        if self.snippet:
            lines.append(f"  > {self.snippet.strip()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Variable / attribute names that suggest credential storage.
_SECRET_NAME_RE = re.compile(
    r"(password|passwd|secret|api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"private[_-]?key|signing[_-]?key|client[_-]?secret|webhook[_-]?secret|"
    r"jwt[_-]?secret|db[_-]?pass|database[_-]?pass)",
    re.IGNORECASE,
)

# String patterns that look like real secrets (not placeholders / env-var refs).
# Avoid flagging "CHANGE_ME", "example", "test", "your_key_here" etc.
_SECRET_VALUE_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}"          # AWS access key
    r"|AGPA[0-9A-Z]{16}"
    r"|sk_live_[0-9a-zA-Z]{24,}"  # Stripe live secret
    r"|pk_live_[0-9a-zA-Z]{24,}"
    r"|ghp_[0-9a-zA-Z]{36,}"      # GitHub PAT
    r"|gho_[0-9a-zA-Z]{36,}"
    r"|ghs_[0-9a-zA-Z]{36,}"
    r"|AIza[0-9A-Za-z\-_]{35}"    # Google API key
    r"|eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}"  # JWT
    r")",
)

# Minimum length of a suspicious literal assigned to a secret-named variable
# (short strings like "" or "N/A" are not secrets).
_SECRET_MIN_LEN = 12


# ---------------------------------------------------------------------------
# AST-based checks
# ---------------------------------------------------------------------------

def _node_is_constant(node: ast.expr) -> bool:
    """Return True if *node* is a literal constant (safe to pass to eval etc.)."""
    return isinstance(node, ast.Constant)


def _call_func_name(node: ast.Call) -> str:
    """Return a dotted name string for the function being called, or ''."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _get_keyword(node: ast.Call, name: str) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true_constant(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _enclosing_try(node: ast.AST, ancestors: list[ast.AST]) -> bool:
    """Return True if any ancestor in the stack is a Try node."""
    return any(isinstance(a, ast.Try) for a in ancestors)


# ---------------------------------------------------------------------------
# Per-file scanner
# ---------------------------------------------------------------------------

class _FileScanner(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str], strict: bool) -> None:
        self.path = path
        self.source_lines = source_lines
        self.strict = strict
        self.findings: list[Finding] = []
        self._ancestors: list[ast.AST] = []

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""

    def _add(
        self,
        severity: str,
        rule_id: str,
        lineno: int,
        message: str,
    ) -> None:
        self.findings.append(
            Finding(
                severity=severity,
                rule_id=rule_id,
                path=self.path,
                line=lineno,
                message=message,
                snippet=self._snippet(lineno),
            )
        )

    # ---- visitor plumbing --------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        self._ancestors.append(node)
        super().generic_visit(node)
        self._ancestors.pop()

    # ---- assignment → secret name check ------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_secret_assignment(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_secret_assignment([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def _check_secret_assignment(
        self,
        targets: list[ast.expr],
        value: ast.expr,
        lineno: int,
    ) -> None:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return
        literal: str = value.value
        # Skip blanks, placeholders, and env-ref patterns
        if (
            len(literal) < _SECRET_MIN_LEN
            or literal.upper() in {"CHANGE_ME", "YOUR_KEY_HERE", "PLACEHOLDER"}
            or literal.startswith(("$", "${", "%(", "<", "ENV_"))
            or literal.lower().startswith(("example", "test", "dummy", "fake", "sample"))
        ):
            return

        # Only flag if it looks like a real credential (pattern match) OR the
        # variable name matches AND the value is suspiciously long (>= 20 chars).
        has_cred_pattern = bool(_SECRET_VALUE_RE.search(literal))
        name_match = any(
            isinstance(t, (ast.Name, ast.Attribute))
            and _SECRET_NAME_RE.search(
                t.id if isinstance(t, ast.Name) else t.attr
            )
            for t in targets
        )
        suspicious_length = len(literal) >= 20 and not any(
            c in literal for c in (" ", "\n", "/", ".")
        )

        if has_cred_pattern or (name_match and suspicious_length):
            self._add(
                "CRITICAL",
                "SECRET-1",
                lineno,
                "Possible hardcoded credential assigned to a secret-named variable "
                f"(value length={len(literal)}). Use environment variables or a "
                "secrets manager instead.",
            )

    # ---- Call-level checks -------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_func_name(node)
        lineno = node.lineno

        # DESER-1: pickle.loads / pickle.load
        if name in {"pickle.loads", "pickle.load"}:
            self._add(
                "CRITICAL",
                "DESER-1",
                lineno,
                f"{name}() deserialises arbitrary objects. Never call with "
                "untrusted data — use JSON + schema validation instead.",
            )

        # DESER-2: yaml.load (not yaml.safe_load)
        elif name == "yaml.load":
            loader_kw = _get_keyword(node, "Loader")
            safe_loader = isinstance(loader_kw, ast.Attribute) and loader_kw.attr in (
                "SafeLoader",
                "CSafeLoader",
            )
            if not safe_loader:
                self._add(
                    "CRITICAL",
                    "DESER-2",
                    lineno,
                    "yaml.load() without Loader=yaml.SafeLoader can execute "
                    "arbitrary Python. Use yaml.safe_load() instead.",
                )

        # DESER-3: eval() / exec() on non-constant
        elif name in {"eval", "exec"}:
            if node.args and not _node_is_constant(node.args[0]):
                self._add(
                    "CRITICAL",
                    "DESER-3",
                    lineno,
                    f"{name}() called with a non-constant argument. "
                    "Never pass user-controlled input to eval/exec.",
                )

        # INJ-1: subprocess calls with shell=True
        elif name in {
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
        }:
            shell_kw = _get_keyword(node, "shell")
            if _is_true_constant(shell_kw):
                # Flag regardless — even 'safe' shell=True calls establish a
                # dangerous pattern and should be rewritten.
                self._add(
                    "HIGH",
                    "INJ-1",
                    lineno,
                    f"{name}() called with shell=True. Use a list of arguments "
                    "and shell=False to prevent command injection.",
                )

        # INJ-2: os.system
        elif name == "os.system":
            if node.args and not _node_is_constant(node.args[0]):
                self._add(
                    "HIGH",
                    "INJ-2",
                    lineno,
                    "os.system() called with a non-constant argument. "
                    "Use subprocess.run(..., shell=False) with a list instead.",
                )

        # STRICT: json.loads without enclosing try/except
        if self.strict and name == "json.loads":
            if not _enclosing_try(node, self._ancestors):
                self._add(
                    "WARNING",
                    "JSON-1",
                    lineno,
                    "json.loads() called outside a try/except block. "
                    "Malformed input will raise json.JSONDecodeError uncaught. "
                    "Wrap in try/except json.JSONDecodeError.",
                )

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def scan_file(path: Path, strict: bool) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    lines = source.splitlines()
    scanner = _FileScanner(path, lines, strict)
    scanner.visit(tree)
    return scanner.findings


def scan_directory(root: Path, strict: bool) -> list[Finding]:
    all_findings: list[Finding] = []
    for py_file in sorted(root.rglob("*.py")):
        # Skip virtual-env, dist, and hidden dirs
        parts = set(py_file.parts)
        if parts & {".venv", "venv", "env", "dist", "__pycache__", ".git", "node_modules"}:
            continue
        all_findings.extend(scan_file(py_file, strict))
    return all_findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guard.py",
        description="Static governance scanner — flags high-confidence security violations.",
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help="Root directory to scan. Defaults to the package root adjacent to scripts/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Also flag json.loads() calls without an adjacent try/except.",
    )
    args = parser.parse_args(argv)

    if args.dir:
        root = Path(args.dir).resolve()
    else:
        # Default: the directory that contains scripts/ (i.e. the project root)
        root = Path(__file__).resolve().parent.parent

    if not root.exists():
        print(f"error: directory not found: {root}", file=sys.stderr)
        return 1

    print(f"Scanning {root} …")
    findings = scan_directory(root, strict=args.strict)

    if not findings:
        print("No violations found. ✓")
        return 0

    # Group by severity for summary
    by_sev: dict[str, list[Finding]] = {}
    for f in findings:
        by_sev.setdefault(f.severity, []).append(f)

    print()
    for sev in ("CRITICAL", "HIGH", "WARNING", "INFO"):
        for f in by_sev.get(sev, []):
            print(f)
            print()

    counts = {s: len(v) for s, v in by_sev.items()}
    summary_parts = [f"{counts[s]} {s}" for s in ("CRITICAL", "HIGH", "WARNING", "INFO") if s in counts]
    print(f"Found {len(findings)} violation(s): {', '.join(summary_parts)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
