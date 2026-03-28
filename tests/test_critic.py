"""Tests for research.critic — pre-commit experiment gate."""

from research.critic import (
    check_artifact_size,
    check_diff_size,
    check_similarity_to_failed,
    merge_verdicts,
)


class TestCheckArtifactSize:
    def test_pass_when_under_limit(self):
        result = check_artifact_size(artifact_bytes=15_000_000)
        assert result["result"] == "pass"

    def test_warn_when_close_to_limit(self):
        result = check_artifact_size(artifact_bytes=15_700_000)
        assert result["result"] == "warn"

    def test_block_when_over_limit(self):
        result = check_artifact_size(artifact_bytes=16_000_001)
        assert result["result"] == "block"

    def test_block_at_soft_limit(self):
        result = check_artifact_size(artifact_bytes=15_800_001)
        assert result["result"] == "block"


class TestCheckDiffSize:
    def test_pass_when_under_100_lines(self):
        result = check_diff_size(lines_changed=50)
        assert result["result"] == "pass"

    def test_warn_when_over_100_lines(self):
        result = check_diff_size(lines_changed=150)
        assert result["result"] == "warn"

    def test_pass_at_exactly_100(self):
        result = check_diff_size(lines_changed=100)
        assert result["result"] == "pass"


class TestCheckSimilarityToFailed:
    def test_no_match_returns_pass(self):
        failed = [
            {"description": "try int3 QAT with symmetric quantization"},
            {"description": "add Mamba layers to transformer"},
        ]
        result = check_similarity_to_failed(
            diff_summary="implement test-time training with context window",
            failed_experiments=failed,
        )
        assert result["result"] == "pass"

    def test_keyword_overlap_returns_warn(self):
        failed = [{"description": "try int3 QAT with symmetric quantization"}]
        result = check_similarity_to_failed(
            diff_summary="implement int3 QAT with asymmetric quantization",
            failed_experiments=failed,
        )
        assert result["result"] == "warn"

    def test_empty_failed_returns_pass(self):
        result = check_similarity_to_failed(diff_summary="anything", failed_experiments=[])
        assert result["result"] == "pass"


class TestMergeVerdicts:
    def test_all_pass(self):
        checks = [
            {"check": "a", "result": "pass", "detail": ""},
            {"check": "b", "result": "pass", "detail": ""},
        ]
        assert merge_verdicts(checks) == "pass"

    def test_warn_overrides_pass(self):
        checks = [
            {"check": "a", "result": "pass", "detail": ""},
            {"check": "b", "result": "warn", "detail": "reason"},
        ]
        assert merge_verdicts(checks) == "warn"

    def test_block_overrides_warn(self):
        checks = [
            {"check": "a", "result": "warn", "detail": ""},
            {"check": "b", "result": "block", "detail": "reason"},
        ]
        assert merge_verdicts(checks) == "block"
