"""Tests for compute.constraints — mathematical feasibility calculator."""

import pytest
from compute.constraints import (
    max_parameters,
    artifact_size,
    quantization_mse,
    training_steps,
    entropy_lower_bound,
    feasibility_report,
    ARTIFACT_LIMIT,
    ARTIFACT_SOFT_LIMIT,
)


class TestMaxParameters:
    def test_int6_fits_reasonable_count(self):
        # 15.8MB at 6-bit with 0.90 compression: ~23.5M params
        params = max_parameters(bits=6, compression_ratio=0.90)
        assert 20_000_000 < params < 30_000_000

    def test_int8_fits_fewer_than_int6(self):
        params_8 = max_parameters(bits=8, compression_ratio=0.90)
        params_6 = max_parameters(bits=6, compression_ratio=0.90)
        assert params_8 < params_6

    def test_code_bytes_reduce_budget(self):
        full = max_parameters(bits=6, code_bytes=0, compression_ratio=0.90)
        with_code = max_parameters(bits=6, code_bytes=50_000, compression_ratio=0.90)
        assert with_code < full
        # 50KB code = ~50K fewer bytes = noticeable but not huge difference
        assert full - with_code > 0

    def test_zero_budget_returns_zero(self):
        assert max_parameters(bits=6, artifact_budget=0, compression_ratio=0.90) == 0

    def test_code_exceeds_budget_returns_zero(self):
        assert max_parameters(bits=6, artifact_budget=1000, code_bytes=2000, compression_ratio=0.90) == 0


class TestArtifactSize:
    def test_inverse_of_max_parameters(self):
        # If max_parameters says N params fit, artifact_size(N) should be near the budget
        params = max_parameters(bits=6, artifact_budget=ARTIFACT_SOFT_LIMIT, code_bytes=0, compression_ratio=0.90)
        size = artifact_size(params, bits=6, code_bytes=0, compression_ratio=0.90)
        # Should be close to the soft limit (within rounding)
        assert abs(size - ARTIFACT_SOFT_LIMIT) < 100

    def test_more_bits_means_larger_artifact(self):
        size_6 = artifact_size(20_000_000, bits=6, compression_ratio=0.90)
        size_8 = artifact_size(20_000_000, bits=8, compression_ratio=0.90)
        assert size_8 > size_6

    def test_more_params_means_larger_artifact(self):
        small = artifact_size(10_000_000, bits=6, compression_ratio=0.90)
        large = artifact_size(20_000_000, bits=6, compression_ratio=0.90)
        assert large > small

    def test_includes_code_bytes(self):
        without = artifact_size(20_000_000, bits=6, code_bytes=0, compression_ratio=0.90)
        with_code = artifact_size(20_000_000, bits=6, code_bytes=50_000, compression_ratio=0.90)
        assert with_code == without + 50_000


class TestQuantizationMSE:
    def test_more_bits_means_lower_mse(self):
        mse_4 = quantization_mse(bits=4, weight_std=0.02)
        mse_6 = quantization_mse(bits=6, weight_std=0.02)
        mse_8 = quantization_mse(bits=8, weight_std=0.02)
        assert mse_4 > mse_6 > mse_8

    def test_asymmetric_lower_than_symmetric(self):
        sym = quantization_mse(bits=6, symmetric=True, weight_std=0.02)
        asym = quantization_mse(bits=6, symmetric=False, weight_std=0.02)
        assert asym < sym

    def test_higher_std_means_higher_mse(self):
        low = quantization_mse(bits=6, weight_std=0.01)
        high = quantization_mse(bits=6, weight_std=0.05)
        assert high > low

    def test_int6_mse_is_small(self):
        mse = quantization_mse(bits=6, weight_std=0.02)
        # int6 should produce very small MSE
        assert mse < 1e-4


class TestTrainingSteps:
    def test_more_params_means_fewer_steps(self):
        small = training_steps(batch_size=64, seq_len=512, model_params=10_000_000)
        large = training_steps(batch_size=64, seq_len=512, model_params=100_000_000)
        assert small > large

    def test_larger_batch_means_fewer_steps(self):
        small_batch = training_steps(batch_size=32, seq_len=512, model_params=20_000_000)
        large_batch = training_steps(batch_size=128, seq_len=512, model_params=20_000_000)
        assert small_batch > large_batch

    def test_returns_positive_for_reasonable_config(self):
        steps = training_steps(batch_size=64, seq_len=512, model_params=20_000_000)
        assert steps > 0

    def test_zero_batch_returns_zero(self):
        assert training_steps(batch_size=0, seq_len=512, model_params=20_000_000) == 0


class TestEntropyLowerBound:
    def test_more_bits_means_higher_bound(self):
        low = entropy_lower_bound(bits=4, params=20_000_000)
        high = entropy_lower_bound(bits=8, params=20_000_000)
        assert high > low

    def test_more_params_means_higher_bound(self):
        small = entropy_lower_bound(bits=6, params=10_000_000)
        large = entropy_lower_bound(bits=6, params=20_000_000)
        assert large > small

    def test_int6_20m_fits_in_16mb(self):
        bound = entropy_lower_bound(bits=6, params=20_000_000)
        assert bound < ARTIFACT_LIMIT


class TestFeasibilityReport:
    def test_feasible_config(self):
        report = feasibility_report(
            params=20_000_000, bits=6, code_bytes=30_000,
        )
        assert report["feasible"] is True
        assert report["checks"]["artifact"]["status"] != "fail"

    def test_infeasible_too_many_params(self):
        report = feasibility_report(
            params=200_000_000, bits=8, code_bytes=30_000,
        )
        assert report["feasible"] is False
        assert report["checks"]["artifact"]["status"] == "fail"

    def test_report_contains_all_checks(self):
        report = feasibility_report(params=20_000_000, bits=6)
        assert "artifact" in report["checks"]
        assert "training_steps" in report["checks"]
        assert "quantization_mse" in report["checks"]
        assert "entropy_bound" in report["checks"]

    def test_max_params_included(self):
        report = feasibility_report(params=20_000_000, bits=6)
        assert "max_params_at_bits" in report
        assert report["max_params_at_bits"] > 0

    def test_compression_ratio_included(self):
        report = feasibility_report(params=20_000_000, bits=6)
        assert "compression_ratio" in report
        assert 0 < report["compression_ratio"] <= 1.0


class TestMemoryFootprint:
    def test_small_model_fits(self):
        from compute.constraints import memory_footprint_check
        result = memory_footprint_check(
            params=20_000_000, bits=6, batch_size=64, seq_len=512
        )
        assert result["status"] == "pass"

    def test_huge_model_fails(self):
        from compute.constraints import memory_footprint_check
        result = memory_footprint_check(
            params=50_000_000_000, bits=16, batch_size=256, seq_len=2048
        )
        assert result["status"] == "fail"

    def test_included_in_feasibility_report(self):
        from compute.constraints import feasibility_report
        report = feasibility_report(params=20_000_000, bits=6)
        assert "memory_footprint" in report["checks"]
        assert report["checks"]["memory_footprint"]["status"] == "pass"
