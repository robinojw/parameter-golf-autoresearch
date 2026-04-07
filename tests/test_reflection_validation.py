"""Tests for reflection validation against results.tsv."""

from dataclasses import dataclass

from research.reflect import _validate_reflection_against_results


@dataclass
class _MockRow:
    description: str
    status: str


class TestReflectionValidation:
    def test_dead_end_with_matching_failure(self):
        rows = [_MockRow(description="try muon optimizer", status="discard")]
        parsed = {
            "technique_updates": [{"node": "muon", "status": "dead_end"}],
            "promising_dimensions": [],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert result["validation_warnings"] == []

    def test_dead_end_without_matching_failure(self):
        rows = [_MockRow(description="try ema averaging", status="keep")]
        parsed = {
            "technique_updates": [{"node": "muon", "status": "dead_end"}],
            "promising_dimensions": [],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert len(result["validation_warnings"]) == 1
        assert "dead_end" in result["validation_warnings"][0]
        assert "muon" in result["validation_warnings"][0]

    def test_proven_with_matching_keep(self):
        rows = [_MockRow(description="try ema averaging", status="keep")]
        parsed = {
            "technique_updates": [{"node": "ema", "status": "proven"}],
            "promising_dimensions": [],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert result["validation_warnings"] == []

    def test_proven_without_matching_keep(self):
        rows = [_MockRow(description="try ema averaging", status="discard")]
        parsed = {
            "technique_updates": [{"node": "slot", "status": "proven"}],
            "promising_dimensions": [],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert len(result["validation_warnings"]) == 1
        assert "proven" in result["validation_warnings"][0]

    def test_promising_dimension_with_multiple_failures(self):
        rows = [
            _MockRow(description="try p2 focal loss v1", status="discard"),
            _MockRow(description="try p2 focal loss v2", status="discard"),
            _MockRow(description="try p2 focal loss v3", status="crash"),
        ]
        parsed = {
            "technique_updates": [],
            "promising_dimensions": ["p2 focal loss improvements"],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert len(result["validation_warnings"]) >= 1
        assert "exhausted" in result["validation_warnings"][0]

    def test_no_warnings_with_empty_results(self):
        parsed = {
            "technique_updates": [{"node": "muon", "status": "dead_end"}],
            "promising_dimensions": ["new direction"],
        }
        result = _validate_reflection_against_results(parsed, [])
        # No warnings because there are no results to validate against
        assert result["validation_warnings"] == []

    def test_empty_node_skipped(self):
        rows = [_MockRow(description="try something", status="discard")]
        parsed = {
            "technique_updates": [{"node": "", "status": "dead_end"}],
            "promising_dimensions": [],
        }
        result = _validate_reflection_against_results(parsed, rows)
        assert result["validation_warnings"] == []
