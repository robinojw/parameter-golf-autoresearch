# research/extract_params.py
"""Extract parameter count and bit-width from text using regex patterns.

Best-effort extraction — returns None for fields that can't be identified.
Used by the grading pipeline to pre-filter infeasible techniques.
"""

from __future__ import annotations

import re
from typing import Optional


def extract_params(text: str) -> dict:
    """Extract parameter count and bit-width from text.

    Returns:
        {"params": int | None, "bits": int | None}
    """
    return {
        "params": _extract_param_count(text),
        "bits": _extract_bitwidth(text),
    }


# ---------------------------------------------------------------------------
# Parameter count extraction
# ---------------------------------------------------------------------------

# "50M params", "1.3B param", "20M model"
_PARAMS_SHORT = re.compile(
    r"(\d+\.?\d*)\s*([MmBb])\s*(?:param|model)",
    re.IGNORECASE,
)

# "20 million parameters"
_PARAMS_WORD = re.compile(
    r"(\d+\.?\d*)\s*(million|billion)\s*param",
    re.IGNORECASE,
)

_MULTIPLIERS = {
    "m": 1_000_000,
    "b": 1_000_000_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _extract_param_count(text: str) -> Optional[int]:
    for pattern in [_PARAMS_SHORT, _PARAMS_WORD]:
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            multiplier = _MULTIPLIERS.get(unit, 1)
            return int(value * multiplier)
    return None


# ---------------------------------------------------------------------------
# Bit-width extraction
# ---------------------------------------------------------------------------

# "int4", "int6", "int8"
_BITS_INT = re.compile(r"int(\d+)", re.IGNORECASE)

# "6-bit", "4 bit", "8-bit"
_BITS_DASH = re.compile(r"(\d+)\s*-?\s*bit", re.IGNORECASE)

# "W4A8" (weight bits = first number)
_BITS_WA = re.compile(r"W(\d+)A\d+", re.IGNORECASE)

# "fp16", "fp32", "bf16", "float16", "float32", "bfloat16"
_BITS_FLOAT = re.compile(r"(?:fp|bf|float|bfloat)(\d+)", re.IGNORECASE)

_FLOAT_BIT_MAP = {
    "16": 16,
    "32": 32,
    "64": 64,
}


def _extract_bitwidth(text: str) -> Optional[int]:
    for pattern in [_BITS_INT, _BITS_DASH, _BITS_WA]:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    # fp/bf/float formats
    match = _BITS_FLOAT.search(text)
    if match:
        return _FLOAT_BIT_MAP.get(match.group(1), int(match.group(1)))
    return None
