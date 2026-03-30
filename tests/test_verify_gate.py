# tests/test_verify_gate.py
from research.verify import filter_infeasible_candidates
from research.fetch import GradedItem


class TestFilterInfeasibleCandidates:
    def test_rejects_infeasible_from_summary(self):
        """agent_summary mentions 200M params at 16-bit — infeasible."""
        item = GradedItem(
            id="big_model",
            score=12.0,
            tier="A",
            score_breakdown={},
            agent_summary="Proposes training a 200M param model at fp16 precision",
            flags=[],
        )
        result = filter_infeasible_candidates([item])
        assert len(result) == 0

    def test_keeps_feasible(self):
        """agent_summary mentions 20M params at int6 — feasible."""
        item = GradedItem(
            id="small_model",
            score=12.0,
            tier="A",
            score_breakdown={},
            agent_summary="Uses int6 quantization on a 20M param transformer",
            flags=[],
        )
        result = filter_infeasible_candidates([item])
        assert len(result) == 1

    def test_keeps_when_no_params_extractable(self):
        """Can't extract params from summary — pass through."""
        item = GradedItem(
            id="vague",
            score=11.0,
            tier="A",
            score_breakdown={},
            agent_summary="Novel attention mechanism that improves convergence",
            flags=[],
        )
        result = filter_infeasible_candidates([item])
        assert len(result) == 1

    def test_skips_already_prefiltered(self):
        """Items already flagged as prefilter_rejected should be skipped."""
        item = GradedItem(
            id="already_rejected",
            score=0,
            tier="C",
            score_breakdown={},
            agent_summary="Auto-rejected: infeasible constraints",
            flags=["prefilter_rejected"],
        )
        result = filter_infeasible_candidates([item])
        assert len(result) == 0
