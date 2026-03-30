# tests/test_prefilter.py
from research.fetch import RawItem
from research.grade import prefilter_infeasible


class TestPrefilterInfeasible:
    def test_rejects_obviously_infeasible(self):
        """200M params at 16-bit can't fit in 16MB."""
        item = RawItem(
            id="test_big",
            source="test",
            dimension=["quantization"],
            title="Training a 200M param fp16 model",
            abstract="We train a 200M parameter model at 16-bit precision",
            url="http://example.com",
            published_date="2026-01-01",
        )
        result = prefilter_infeasible([item])
        assert len(result["rejected"]) == 1
        assert len(result["passed"]) == 0
        assert result["rejected"][0].id == "test_big"

    def test_passes_feasible(self):
        """20M params at 6-bit fits easily."""
        item = RawItem(
            id="test_small",
            source="test",
            dimension=["quantization"],
            title="Efficient int6 quantization for 20M param models",
            abstract="We apply int6 QAT to a 20M parameter transformer",
            url="http://example.com",
            published_date="2026-01-01",
        )
        result = prefilter_infeasible([item])
        assert len(result["passed"]) == 1
        assert len(result["rejected"]) == 0

    def test_passes_when_no_params_extractable(self):
        """Can't determine params — pass through to LLM grading."""
        item = RawItem(
            id="test_vague",
            source="test",
            dimension=["attention"],
            title="A novel attention mechanism for language models",
            abstract="We propose an efficient attention variant",
            url="http://example.com",
            published_date="2026-01-01",
        )
        result = prefilter_infeasible([item])
        assert len(result["passed"]) == 1
        assert len(result["rejected"]) == 0

    def test_passes_when_only_bits_extractable(self):
        """Only bits found, no params — can't check feasibility, pass through."""
        item = RawItem(
            id="test_bits_only",
            source="test",
            dimension=["quantization"],
            title="Novel int4 quantization scheme",
            abstract="We propose a new 4-bit quantization method",
            url="http://example.com",
            published_date="2026-01-01",
        )
        result = prefilter_infeasible([item])
        assert len(result["passed"]) == 1
        assert len(result["rejected"]) == 0
