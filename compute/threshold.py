"""Dynamic promotion threshold — scales required improvement with distance from SOTA."""

from __future__ import annotations

from compute.constraints import ARTIFACT_LIMIT, ARTIFACT_SOFT_LIMIT, TRAINING_TIME_LIMIT

_DEFAULT_BASELINE = 1.2244
_DEFAULT_MIN_REQUIRED = 0.005
_DEFAULT_MAX_REQUIRED = 0.015
_DEFAULT_FALLBACK_WINDOW = 10


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def compute_promotion_threshold(
    current_bpb: float,
    sota: float,
    baseline: float = _DEFAULT_BASELINE,
    min_required: float = _DEFAULT_MIN_REQUIRED,
    max_required: float = _DEFAULT_MAX_REQUIRED,
) -> float:
    """Return threshold ratio. Caller checks: candidate_bpb < current_bpb * threshold."""
    if baseline <= sota:
        return 1.0 - min_required
    progress = _clamp((current_bpb - sota) / (baseline - sota), 0.0, 1.0)
    required = _lerp(min_required, max_required, progress)
    return 1.0 - required


def check_adaptive_fallback(
    recent_rows: list[dict],
    current_bpb: float,
    computed_threshold: float,
    window: int = _DEFAULT_FALLBACK_WINDOW,
) -> float | None:
    """Check if threshold should be relaxed based on recent history.

    Returns a relaxed threshold ratio, or None if no relaxation needed.
    """
    local_keeps = [
        r for r in recent_rows
        if r.get("tier", "").lower() == "local"
        and r.get("status", "").lower() == "keep"
        and _safe_float(r.get("val_bpb")) > 0
    ]
    tail = local_keeps[-window:]
    if not tail:
        return None
    best_bpb = min(_safe_float(r["val_bpb"]) for r in tail)
    is_improvement = best_bpb < current_bpb
    already_passes = best_bpb < current_bpb * computed_threshold
    if not is_improvement or already_passes:
        return None
    relaxed = (best_bpb / current_bpb) + 0.001
    return relaxed


def pareto_score(
    candidate_bpb: float,
    candidate_artifact_bytes: int,
    candidate_training_seconds: float,
    current_bpb: float,
    sota: float,
    baseline: float = _DEFAULT_BASELINE,
) -> dict:
    """Score a candidate across bpb, artifact headroom, and time headroom.

    Returns dict with per-axis scores, dynamic weights, weighted total,
    and a promote flag (True only if bpb improves AND hard constraints pass).
    """
    # --- bpb axis (fractional improvement, higher = better) ---
    bpb_delta = (current_bpb - candidate_bpb) / current_bpb

    # --- artifact headroom (fraction of budget remaining) ---
    artifact_headroom = _clamp(
        1.0 - (candidate_artifact_bytes / ARTIFACT_LIMIT), 0.0, 1.0,
    )

    # --- time headroom (fraction of budget remaining) ---
    time_headroom = _clamp(
        1.0 - (candidate_training_seconds / TRAINING_TIME_LIMIT), 0.0, 1.0,
    )

    # --- dynamic weights: shift from bpb toward headroom as we approach SOTA ---
    if baseline <= sota:
        progress = 0.0
    else:
        progress = _clamp((current_bpb - sota) / (baseline - sota), 0.0, 1.0)

    # progress=1 → far from SOTA (near baseline): weight bpb heavily
    # progress=0 → close to SOTA: weight headroom more
    w_bpb = _lerp(0.50, 0.80, progress)
    w_artifact = _lerp(0.30, 0.10, progress)
    w_time = _lerp(0.20, 0.10, progress)

    total = w_bpb * bpb_delta + w_artifact * artifact_headroom + w_time * time_headroom

    # --- gates ---
    bpb_passes = bpb_delta > 0
    artifact_ok = candidate_artifact_bytes < ARTIFACT_SOFT_LIMIT
    time_ok = candidate_training_seconds < TRAINING_TIME_LIMIT

    return {
        "bpb_delta": bpb_delta,
        "artifact_headroom": artifact_headroom,
        "time_headroom": time_headroom,
        "weights": {"bpb": w_bpb, "artifact": w_artifact, "time": w_time},
        "score": total,
        "promote": bpb_passes and artifact_ok and time_ok,
    }


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
