"""Tests for compute.threshold — dynamic promotion threshold."""

import pytest
from compute.threshold import compute_promotion_threshold, check_adaptive_fallback

SOTA = 1.1194
BASELINE = 1.2244


class TestComputePromotionThreshold:
    def test_far_from_sota_requires_larger_improvement(self):
        threshold = compute_promotion_threshold(current_bpb=1.22, sota=SOTA, baseline=BASELINE)
        assert 0.984 < threshold < 0.986

    def test_near_sota_requires_smaller_improvement(self):
        threshold = compute_promotion_threshold(current_bpb=1.12, sota=SOTA, baseline=BASELINE)
        assert 0.994 < threshold < 0.996

    def test_at_sota_uses_min_required(self):
        threshold = compute_promotion_threshold(current_bpb=SOTA, sota=SOTA, baseline=BASELINE)
        assert threshold == pytest.approx(1.0 - 0.005, abs=0.001)

    def test_below_sota_uses_min_required(self):
        threshold = compute_promotion_threshold(current_bpb=1.10, sota=SOTA, baseline=BASELINE)
        assert threshold == pytest.approx(1.0 - 0.005, abs=0.001)

    def test_baseline_equals_sota_uses_min(self):
        threshold = compute_promotion_threshold(current_bpb=1.12, sota=1.12, baseline=1.12)
        assert threshold == pytest.approx(1.0 - 0.005, abs=0.001)

    def test_monotonic_threshold_relaxes_as_bpb_approaches_sota(self):
        thresholds = [
            compute_promotion_threshold(bpb, SOTA, BASELINE)
            for bpb in [1.22, 1.20, 1.18, 1.16, 1.14, 1.12]
        ]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] < thresholds[i + 1]


class TestCheckAdaptiveFallback:
    def test_returns_none_when_no_recent_keeps(self):
        result = check_adaptive_fallback([], current_bpb=1.15, computed_threshold=0.99, window=10)
        assert result is None

    def test_returns_none_when_best_is_not_an_improvement(self):
        rows = [
            {"tier": "local", "status": "keep", "val_bpb": 1.16},
            {"tier": "local", "status": "keep", "val_bpb": 1.17},
        ]
        result = check_adaptive_fallback(rows, current_bpb=1.15, computed_threshold=0.99, window=10)
        assert result is None

    def test_returns_relaxed_threshold_when_best_improves(self):
        rows = [
            {"tier": "local", "status": "keep", "val_bpb": 1.148},
            {"tier": "local", "status": "keep", "val_bpb": 1.155},
        ]
        result = check_adaptive_fallback(rows, current_bpb=1.15, computed_threshold=0.99, window=10)
        assert result is not None
        assert 1.148 < 1.15 * result

    def test_respects_window_size(self):
        old_good = [{"tier": "local", "status": "keep", "val_bpb": 1.10}]
        recent_bad = [{"tier": "local", "status": "keep", "val_bpb": 1.16} for _ in range(10)]
        rows = old_good + recent_bad
        result = check_adaptive_fallback(rows, current_bpb=1.15, computed_threshold=0.99, window=10)
        assert result is None

    def test_ignores_non_local_and_non_keep_rows(self):
        rows = [
            {"tier": "runpod", "status": "keep", "val_bpb": 1.10},
            {"tier": "local", "status": "discard", "val_bpb": 1.10},
            {"tier": "local", "status": "keep", "val_bpb": 1.16},
        ]
        result = check_adaptive_fallback(rows, current_bpb=1.15, computed_threshold=0.99, window=10)
        assert result is None
