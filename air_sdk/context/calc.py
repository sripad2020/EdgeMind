"""Safe arithmetic: compute the expected answer of a plain calculation question.

"What is 17 * 23?", "calculate (12 + 3) x 4", "15% of 840", "1250 divided by 25".

Security: the expression is parsed with ``ast`` and only number literals, + - * / // %
and ** with a small exponent are allowed. No names, calls, attributes or comprehensions,
so nothing can be executed. Length, exponent and magnitude are bounded.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Final

MAX_EXPR_CHARS: Final[int] = 200
MAX_EXPONENT: Final[float] = 64
MAX_MAGNITUDE: Final[float] = 1e100

_WORDS: Final = (
    (r"\bmultiplied by\b|\btimes\b|\u00d7|\bx\b(?=\s*[\d(])", "*"),  # regex \u escapes
    (r"\bdivided by\b|\u00f7", "/"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bto the power of\b|\^", "**"),
)
_PERCENT_OF: Final = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I)
_PERCENT: Final = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CANDIDATE: Final = re.compile(r"[\d(][\d\s.+\-*/%()]*[\d)]")

_BIN: Final[dict[type[ast.operator], Callable[[float, float], float]]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}


class _UnsafeError(Exception):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise _UnsafeError("exponent too large")
            power: object = left ** right
            if not isinstance(power, float):  # e.g. complex for (-8) ** 0.5
                raise _UnsafeError("result is not a real number")
            result = power
        else:
            fn = _BIN.get(type(node.op))
            if fn is None:
                raise _UnsafeError("operator not allowed")
            result = fn(left, right)
        if not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
            raise _UnsafeError("result out of range")
        return result
    raise _UnsafeError(f"{type(node).__name__} not allowed")


def evaluate(expression: str) -> float | None:
    """Evaluate a plain arithmetic expression safely; None if it is not one."""
    if not expression or len(expression) > MAX_EXPR_CHARS:
        return None
    try:
        return _eval(ast.parse(expression.strip(), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError,
            _UnsafeError):
        return None


def expected_answer(question: str) -> tuple[str, float] | None:
    """Find a calculation in a question and compute it: returns (expression, value)."""
    text = question[:2000]
    for pattern, symbol in _WORDS:
        text = re.sub(pattern, symbol, text, flags=re.I)
    text = _PERCENT_OF.sub(lambda m: f"({m.group(1)}/100*{m.group(2)})", text)
    text = _PERCENT.sub(lambda m: f"({m.group(1)}/100)", text)
    best: tuple[str, float] | None = None
    for m in _CANDIDATE.finditer(text):
        expr = m.group(0).strip()
        if not re.search(r"[+\-*/%]", expr.lstrip("-")) or len(expr) > MAX_EXPR_CHARS:
            continue  # a lone number is not a calculation
        value = evaluate(expr)
        if value is not None and (best is None or len(expr) > len(best[0])):
            best = (expr, value)
    return best


def format_number(value: float) -> str:
    """Compact text for a computed value (no trailing .0, limited decimals)."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"
