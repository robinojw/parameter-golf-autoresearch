"""Extract structured quantitative claims from research text.

Deterministic regex-based extraction of claims like:
  "+3.2% accuracy on ImageNet with 50M params"
  "reduces perplexity by 0.05 on WikiText-103"
  "1.08 bpb on FineWeb validation"

Claims are structured as:
  {technique, metric, magnitude, direction, condition, raw_text}

This gives the LLM grader a much tighter target than free-form abstract text,
reducing hallucination surface area per the Perplexity feedback analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Claim:
    """A single quantitative claim extracted from text."""

    technique: str  # what was applied
    metric: str  # what was measured
    magnitude: str  # the number (e.g. "3.2%", "0.05", "1.08")
    direction: str  # "improvement", "reduction", "absolute", "unknown"
    condition: str  # context (e.g. "on ImageNet", "with 50M params", "at 4-bit")
    raw_text: str  # the original matched text span

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric patterns — what was measured
# ---------------------------------------------------------------------------
_METRIC_NAMES = (
    r"(?:val[_\s]?)?bpb|bits?\s*per\s*byte|perplexity|ppl|accuracy|"
    r"f1[\s\-]?score|bleu|rouge|loss|val[_\s]?loss|"
    r"throughput|tokens?/s|samples?/s|wall[\s\-]?clock|"
    r"compression\s*ratio|artifact\s*size"
)

# ---------------------------------------------------------------------------
# Magnitude patterns — the number
# ---------------------------------------------------------------------------
# "+3.2%", "-0.05", "0.05 bpb", "by 15%", "to 1.08"
_MAGNITUDE_PATTERN = re.compile(
    r"(?:by\s+|to\s+|of\s+|=\s*|:\s*)?"
    r"([+-]?\d+\.?\d*)\s*(%|pp|percentage\s*points?|bpb|bits?\s*per\s*byte)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Direction patterns
# ---------------------------------------------------------------------------
_IMPROVEMENT_WORDS = frozenset({
    "improve", "improves", "improved", "improvement", "gain", "gains",
    "increase", "increases", "increased", "boost", "boosts", "boosted",
    "better", "outperform", "outperforms", "surpass", "surpasses",
    "achieve", "achieves", "achieved", "higher",
})
_REDUCTION_WORDS = frozenset({
    "reduce", "reduces", "reduced", "reduction", "decrease", "decreases",
    "decreased", "lower", "lowers", "lowered", "drop", "drops", "dropped",
    "cut", "cuts", "minimize", "minimizes", "minimized", "shrink", "shrinks",
})

# ---------------------------------------------------------------------------
# Full claim patterns (ordered by specificity)
# ---------------------------------------------------------------------------

# Pattern 1: "achieves X.XX bpb on Y" / "X.XX val_bpb"
_CLAIM_ABSOLUTE = re.compile(
    r"(?:achiev\w+|reach\w+|obtain\w+|report\w+|get\w+)?\s*"
    r"(\d+\.?\d*)\s*"
    r"(" + _METRIC_NAMES + r")"
    r"(?:\s+(?:on|for|with|at)\s+(.+?))?(?:\.|,|;|$)",
    re.IGNORECASE,
)

# Pattern 2: "improves/reduces X by Y%" / "+Y% on metric"
_CLAIM_DELTA = re.compile(
    r"(?:(" + "|".join(_IMPROVEMENT_WORDS | _REDUCTION_WORDS) + r")\w*\s+)?"
    r"(?:the\s+)?(?:" + _METRIC_NAMES + r")?\s*"
    r"(?:by\s+)?([+-]?\d+\.?\d*)\s*(%|pp|bpb|bits?\s*per\s*byte)"
    r"(?:\s+(?:on|for|with|over|compared\s+to)\s+(.+?))?(?:\.|,|;|$)",
    re.IGNORECASE,
)

# Pattern 3: "X% improvement/reduction in Y"
_CLAIM_PERCENT = re.compile(
    r"(\d+\.?\d*)\s*(%|pp)\s+"
    r"(improvement|reduction|gain|drop|decrease|increase)\s+"
    r"(?:in|on|of|for)\s+(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)

# Pattern 4: "from X to Y bpb" / "X → Y"
_CLAIM_RANGE = re.compile(
    r"(?:from\s+)?(\d+\.?\d*)\s*(?:→|->|to)\s*(\d+\.?\d*)\s*"
    r"(" + _METRIC_NAMES + r")"
    r"(?:\s+(?:on|for|with)\s+(.+?))?(?:\.|,|;|$)",
    re.IGNORECASE,
)


def _infer_direction(text: str) -> str:
    """Infer claim direction from surrounding words."""
    words = set(text.lower().split())
    if words & _IMPROVEMENT_WORDS:
        return "improvement"
    if words & _REDUCTION_WORDS:
        return "reduction"
    return "unknown"


def _extract_technique_context(text: str, claim_start: int) -> str:
    """Extract technique name from text preceding the claim."""
    prefix = text[:claim_start].strip()
    # Take the last sentence or clause before the claim
    for sep in [". ", "; ", ", "]:
        if sep in prefix:
            prefix = prefix.rsplit(sep, 1)[-1]
    # Truncate to reasonable length
    return prefix[-120:].strip()


def extract_claims(text: str) -> list[Claim]:
    """Extract all quantitative claims from text.

    Returns a list of Claim objects, deduplicated by raw_text.
    Deterministic — no LLM calls.
    """
    claims: list[Claim] = []
    seen_spans: set[str] = set()

    # Pattern 4: range claims (most specific)
    for m in _CLAIM_RANGE.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_spans:
            continue
        seen_spans.add(raw)
        from_val, to_val = float(m.group(1)), float(m.group(2))
        direction = "reduction" if to_val < from_val else "improvement"
        delta = abs(from_val - to_val)
        claims.append(Claim(
            technique=_extract_technique_context(text, m.start()),
            metric=m.group(3).strip(),
            magnitude=f"{delta:.4f}",
            direction=direction,
            condition=m.group(4).strip() if m.group(4) else "",
            raw_text=raw,
        ))

    # Pattern 3: "X% improvement in Y"
    for m in _CLAIM_PERCENT.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_spans:
            continue
        seen_spans.add(raw)
        direction_word = m.group(3).lower()
        direction = "reduction" if direction_word in ("reduction", "drop", "decrease") else "improvement"
        claims.append(Claim(
            technique=_extract_technique_context(text, m.start()),
            metric=m.group(4).strip(),
            magnitude=f"{m.group(1)}{m.group(2)}",
            direction=direction,
            condition="",
            raw_text=raw,
        ))

    # Pattern 2: delta claims
    for m in _CLAIM_DELTA.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_spans:
            continue
        seen_spans.add(raw)
        direction_word = m.group(1) or ""
        direction = _infer_direction(direction_word) if direction_word else "unknown"
        # Check sign of number
        val_str = m.group(2)
        if val_str.startswith("-"):
            direction = "reduction"
        elif val_str.startswith("+"):
            direction = "improvement"
        claims.append(Claim(
            technique=_extract_technique_context(text, m.start()),
            metric="",
            magnitude=f"{val_str}{m.group(3)}",
            direction=direction,
            condition=m.group(4).strip() if m.group(4) else "",
            raw_text=raw,
        ))

    # Pattern 1: absolute claims
    for m in _CLAIM_ABSOLUTE.finditer(text):
        raw = m.group(0).strip()
        if raw in seen_spans:
            continue
        seen_spans.add(raw)
        claims.append(Claim(
            technique=_extract_technique_context(text, m.start()),
            metric=m.group(2).strip(),
            magnitude=m.group(1),
            direction="absolute",
            condition=m.group(3).strip() if m.group(3) else "",
            raw_text=raw,
        ))

    return claims


def format_claims_for_grading(claims: list[Claim]) -> str:
    """Format extracted claims as a compact string for injection into grading prompts.

    Returns empty string if no claims found.
    """
    if not claims:
        return ""
    lines = ["EXTRACTED CLAIMS (deterministic, from abstract):"]
    for i, c in enumerate(claims, 1):
        parts = [f"  {i}. [{c.direction}]"]
        if c.technique:
            parts.append(f"technique={c.technique[:60]}")
        parts.append(f"magnitude={c.magnitude}")
        if c.metric:
            parts.append(f"metric={c.metric}")
        if c.condition:
            parts.append(f"condition={c.condition[:60]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
