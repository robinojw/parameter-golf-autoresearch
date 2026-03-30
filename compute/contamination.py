# compute/contamination.py
"""Deterministic contamination detection for test-time training leakage.

Checks whether training scripts reference validation data in training loops,
and whether score improvements are plausibly explained by training alone.
All checks are code-level — no LLM judgment.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContaminationResult:
    status: str  # "pass", "warn", "block"
    check: str
    detail: str
    references: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST-based data overlap detection
# ---------------------------------------------------------------------------

# Patterns that indicate eval-only usage (not training)
_EVAL_CONTEXT_PATTERNS = [
    "no_grad", "torch.no_grad", "mx.no_grad", "evaluate", "eval_mode",
    "model.eval", "inference_mode",
]

# Patterns that indicate gradient computation (training usage)
# Use specific multi-char patterns to avoid false matches (e.g. "grad" inside "no_grad")
_TRAIN_CONTEXT_PATTERNS = [
    ".backward", ".grad", "optimizer.", "loss.backward", "optimizer.step",
]


def _find_string_references(tree: ast.Module, val_paths: list[str]) -> list[tuple[int, str]]:
    """Find all string literals in the AST that match any val_path."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for vp in val_paths:
                if vp in node.value:
                    hits.append((node.lineno, node.value))
    return hits


def _line_in_eval_context(source_lines: list[str], lineno: int) -> bool:
    """Check if a line number is within an eval-only context.

    Scans a window of 5 lines before and 5 lines after for eval patterns.
    If eval patterns are found and no training patterns are found, it's eval-only.
    """
    start = max(0, lineno - 6)
    end = min(len(source_lines), lineno + 5)
    window = "\n".join(source_lines[start:end]).lower()

    has_eval = any(pat in window for pat in _EVAL_CONTEXT_PATTERNS)
    has_train = any(pat in window for pat in _TRAIN_CONTEXT_PATTERNS)

    return has_eval and not has_train


def check_data_overlap(
    script_path: Path,
    val_paths: list[str] | None = None,
) -> ContaminationResult:
    """Check whether a training script references validation data in training loops."""
    if val_paths is None:
        val_paths = ["val.bin", "val_data", "validation", "fineweb_val"]

    source = script_path.read_text()
    source_lines = source.split("\n")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ContaminationResult(
            status="warn",
            check="data_overlap",
            detail=f"Could not parse {script_path}: syntax error",
        )

    hits = _find_string_references(tree, val_paths)
    if not hits:
        return ContaminationResult(
            status="pass",
            check="data_overlap",
            detail="No validation data references found in script",
        )

    training_refs: list[str] = []
    for lineno, value in hits:
        if not _line_in_eval_context(source_lines, lineno):
            training_refs.append(f"line {lineno}: {value}")

    if not training_refs:
        return ContaminationResult(
            status="pass",
            check="data_overlap",
            detail="Validation data referenced only in eval context",
        )

    return ContaminationResult(
        status="block",
        check="data_overlap",
        detail=f"Validation data referenced outside eval context: {', '.join(training_refs)}",
        references=training_refs,
    )


# ---------------------------------------------------------------------------
# Score plausibility check
# ---------------------------------------------------------------------------

_VAL_TRAIN_RATIO_WARN = 3.0
_VAL_TRAIN_RATIO_BLOCK = 20.0


def check_score_plausibility(
    train_bpb_before: float,
    train_bpb_after: float,
    val_bpb_before: float,
    val_bpb_after: float,
) -> ContaminationResult:
    """Check whether val improvement is plausibly explained by training."""
    train_delta = train_bpb_before - train_bpb_after
    val_delta = val_bpb_before - val_bpb_after

    if val_delta <= 0:
        return ContaminationResult(
            status="pass",
            check="score_plausibility",
            detail=f"No val improvement (delta={val_delta:+.4f})",
        )

    if train_delta < 0 and val_delta > 0.01:
        return ContaminationResult(
            status="block",
            check="score_plausibility",
            detail=f"Val improved ({val_delta:+.4f}) but train worsened ({train_delta:+.4f})",
        )

    if train_delta > 0:
        ratio = val_delta / train_delta
        if ratio > _VAL_TRAIN_RATIO_BLOCK:
            return ContaminationResult(
                status="block",
                check="score_plausibility",
                detail=f"Val/train improvement ratio {ratio:.1f}x exceeds block threshold",
            )
        if ratio > _VAL_TRAIN_RATIO_WARN:
            return ContaminationResult(
                status="warn",
                check="score_plausibility",
                detail=f"Val/train improvement ratio {ratio:.1f}x exceeds warn threshold",
            )

    return ContaminationResult(
        status="pass",
        check="score_plausibility",
        detail=f"Improvement plausible (train={train_delta:+.4f}, val={val_delta:+.4f})",
    )
