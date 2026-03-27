"""Public API for experiment history — read-only, standalone (no circular deps)."""

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

RESULTS_TSV_PATH = Path("results.tsv")
COMPETITOR_SCORES_PATH = Path("competitor_scores.jsonl")

FALLBACK_SOTA: float = 1.1194
BASELINE_BPB: float = 1.2244

BASELINE_PROVEN_TECHNIQUES: list[str] = [
    "int6 QAT",
    "zstd-22",
    "EMA",
    "BigramHash",
    "sliding window eval",
    "partial RoPE",
    "XSA attention",
    "SmearGate",
    "OrthoInit",
    "LeakyReLU²",
    "Parallel Muon + AdamW",
    "ternary quantization",
]

_COL_COMMIT = "commit"
_COL_TIER = "tier"
_COL_VAL_BPB = "val_bpb"
_COL_ARTIFACT_BYTES = "artifact_bytes"
_COL_MEMORY_GB = "memory_gb"
_COL_STATUS = "status"
_COL_PROMOTED = "promoted"
_COL_COST_USD = "cost_usd"
_COL_DESCRIPTION = "description"
_KEY_PAIRS = "pairs"
_KEY_AVG_DELTA = "avg_delta"
_KEY_CORRELATION_RELIABLE = "correlation_reliable"
_KEY_PR_NUMBER = "pr_number"
_KEY_AUTHOR = "author"
_KEY_TITLE = "title"
_KEY_TECHNIQUE = "technique"
_KEY_DELTA_FROM_BASELINE = "delta_from_baseline"

_TIER_RUNPOD = "runpod"
_TIER_LOCAL = "local"
_STATUS_KEEP = "keep"
_STATUS_DISCARD = "discard"
_STATUS_CRASH = "crash"
_PROMOTED_YES = "yes"
_ZERO = 0.0
_TSV_HEADER_OFFSET = 2
_MAX_FAILED = 10
_MIN_CORRELATION_PAIRS = 5
_DELTA_ROUND_DIGITS = 6
_DEFAULT_HISTORY_LIMIT = 8
_TECHNIQUE_PREFIXES = ("try ", "test ", "add ", "implement ", "enable ", "use ")
_CLAUSE_SEPARATORS = (",", ";", " — ", " - ")


@dataclass
class _ExperimentRow:
    commit: str
    tier: str
    val_bpb: float
    artifact_bytes: str
    memory_gb: str
    status: str
    promoted: str
    cost_usd: float
    description: str


def _safe_float(raw_str: str) -> float:
    stripped = raw_str.strip()
    return float(stripped) if stripped else _ZERO


def _safe_str(raw: dict, key: str) -> str:
    return (raw.get(key) or "").strip()


def _parse_single_row(raw: dict) -> _ExperimentRow:
    return _ExperimentRow(
        commit=_safe_str(raw, _COL_COMMIT),
        tier=_safe_str(raw, _COL_TIER),
        val_bpb=_safe_float(_safe_str(raw, _COL_VAL_BPB)),
        artifact_bytes=_safe_str(raw, _COL_ARTIFACT_BYTES),
        memory_gb=_safe_str(raw, _COL_MEMORY_GB),
        status=_safe_str(raw, _COL_STATUS),
        promoted=_safe_str(raw, _COL_PROMOTED).lower(),
        cost_usd=_safe_float(_safe_str(raw, _COL_COST_USD)),
        description=_safe_str(raw, _COL_DESCRIPTION),
    )


def _read_rows() -> list[_ExperimentRow]:
    if not RESULTS_TSV_PATH.exists():
        return []
    rows: list[_ExperimentRow] = []
    try:
        with open(RESULTS_TSV_PATH, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for lineno, raw in enumerate(reader, start=_TSV_HEADER_OFFSET):
                try:
                    rows.append(_parse_single_row(raw))
                except (ValueError, TypeError) as exc:
                    warnings.warn(
                        f"results.tsv line {lineno}: skipping malformed row — {exc}",
                        stacklevel=_TSV_HEADER_OFFSET,
                    )
    except OSError as exc:
        warnings.warn(
            f"Could not read {RESULTS_TSV_PATH}: {exc}", stacklevel=_TSV_HEADER_OFFSET
        )
    return rows


def _extract_technique(description: str) -> str:
    text = description.strip()
    for prefix in _TECHNIQUE_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    for sep in _CLAUSE_SEPARATORS:
        if sep in text:
            text = text[: text.index(sep)]
            break
    return text.strip()


def _description_matches(desc_a: str, desc_b: str) -> bool:
    lower_a = desc_a.lower().strip()
    lower_b = desc_b.lower().strip()
    both_non_empty = bool(lower_a) and bool(lower_b)
    return both_non_empty and (lower_a in lower_b or lower_b in lower_a)


def _is_runpod_keep(row: _ExperimentRow) -> bool:
    return (
        row.tier.lower() == _TIER_RUNPOD
        and row.status.lower() == _STATUS_KEEP
        and row.val_bpb > _ZERO
    )


def _is_promoted_runpod(row: _ExperimentRow) -> bool:
    is_runpod_with_keep = (
        row.tier.lower() == _TIER_RUNPOD and _STATUS_KEEP in row.status.lower()
    )
    return (
        is_runpod_with_keep and row.promoted == _PROMOTED_YES and bool(row.description)
    )


def _technique_already_covered(technique: str, existing: list[str]) -> bool:
    lower_t = technique.lower()
    return any(lower_t in e.lower() or e.lower() in lower_t for e in existing)


def _is_failed(row: _ExperimentRow) -> bool:
    return row.status.lower() in (_STATUS_DISCARD, _STATUS_CRASH)


def get_current_best_bpb() -> float:
    """Lowest val_bpb from tier=runpod, status=keep. Falls back to FALLBACK_SOTA."""
    candidates = [r.val_bpb for r in _read_rows() if _is_runpod_keep(r)]
    return min(candidates) if candidates else FALLBACK_SOTA


def get_proven_techniques() -> list[str]:
    """Technique names from promoted runpod experiments merged with BASELINE list."""
    dynamic: list[str] = []
    for row in _read_rows():
        if not _is_promoted_runpod(row):
            continue
        technique = _extract_technique(row.description)
        is_new_technique = bool(technique) and technique not in dynamic
        if is_new_technique:
            dynamic.append(technique)
    combined = list(BASELINE_PROVEN_TECHNIQUES)
    for technique in dynamic:
        if not _technique_already_covered(technique, combined):
            combined.append(technique)
    return combined


def get_failed_experiments() -> list[dict]:
    """Most recent 10 experiments with status=discard or crash."""
    failed = [
        {_COL_DESCRIPTION: r.description, _COL_VAL_BPB: r.val_bpb, _COL_TIER: r.tier}
        for r in _read_rows()
        if _is_failed(r)
    ]
    return failed[-_MAX_FAILED:]


def get_experiment_history_bullets(limit: int = _DEFAULT_HISTORY_LIMIT) -> str:
    """Markdown bullets of the last N experiments for injection into program.md."""
    rows = _read_rows()
    recent = rows[-limit:] if rows else []
    lines: list[str] = []
    for r in recent:
        lines.append(
            f"- [{r.tier}] {r.description} — val_bpb={r.val_bpb:.4f}, "
            f"status={r.status} (cost=${r.cost_usd:.2f})"
        )
    return "\n".join(lines)


def get_tier_correlation() -> dict:
    """Correlation between local and runpod val_bpb for description-matched pairs."""
    rows = _read_rows()
    local_rows = [
        r for r in rows if r.tier.lower() == _TIER_LOCAL and r.val_bpb > _ZERO
    ]
    runpod_rows = [
        r for r in rows if r.tier.lower() == _TIER_RUNPOD and r.val_bpb > _ZERO
    ]
    pairs: list[tuple[float, float]] = []
    used_runpod: set[int] = set()
    for lr in local_rows:
        for idx, rr in enumerate(runpod_rows):
            if idx in used_runpod:
                continue
            if _description_matches(lr.description, rr.description):
                pairs.append((lr.val_bpb, rr.val_bpb))
                used_runpod.add(idx)
                break
    if not pairs:
        return {_KEY_PAIRS: 0, _KEY_AVG_DELTA: _ZERO, _KEY_CORRELATION_RELIABLE: False}
    deltas = [local_bpb - runpod_bpb for local_bpb, runpod_bpb in pairs]
    avg_delta = sum(deltas) / len(deltas)
    return {
        _KEY_PAIRS: len(pairs),
        _KEY_AVG_DELTA: round(avg_delta, _DELTA_ROUND_DIGITS),
        _KEY_CORRELATION_RELIABLE: len(pairs) >= _MIN_CORRELATION_PAIRS,
    }


def _parse_competitor_entry(obj: dict) -> dict:
    return {
        _KEY_PR_NUMBER: int(obj.get(_KEY_PR_NUMBER, 0)),
        _KEY_AUTHOR: str(obj.get(_KEY_AUTHOR, "")),
        _KEY_TITLE: str(obj.get(_KEY_TITLE, "")),
        _COL_VAL_BPB: float(obj.get(_COL_VAL_BPB, _ZERO)),
        _KEY_TECHNIQUE: str(obj.get(_KEY_TECHNIQUE, "")),
        _KEY_DELTA_FROM_BASELINE: float(obj.get(_KEY_DELTA_FROM_BASELINE, _ZERO)),
    }


def get_competitor_scores() -> list[dict]:
    """Competitor scores from competitor_scores.jsonl, sorted by val_bpb ascending."""
    if not COMPETITOR_SCORES_PATH.exists():
        return []
    entries: list[dict] = []
    try:
        with open(COMPETITOR_SCORES_PATH) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(_parse_competitor_entry(json.loads(line)))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    warnings.warn(
                        f"{COMPETITOR_SCORES_PATH} line {lineno}: skipping — {exc}",
                        stacklevel=_TSV_HEADER_OFFSET,
                    )
    except OSError as exc:
        warnings.warn(
            f"Could not read {COMPETITOR_SCORES_PATH}: {exc}",
            stacklevel=_TSV_HEADER_OFFSET,
        )
    entries.sort(key=lambda entry: entry[_COL_VAL_BPB])
    return entries
