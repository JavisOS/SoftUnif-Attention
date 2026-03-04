"""Safe parsing helpers for CSV-stored Python literals."""

import ast
from typing import Any


def safe_literal_eval(value: Any, default: Any = None) -> Any:
    if not isinstance(value, str):
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def parse_pair_literal(value: Any, default: Any = None):
    parsed = safe_literal_eval(value, default=None)
    if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
        return (parsed[0], parsed[1])
    return default
