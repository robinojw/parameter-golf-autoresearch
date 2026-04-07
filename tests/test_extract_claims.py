"""Tests for research/extract_claims.py — deterministic claim extraction."""

from research.extract_claims import Claim, extract_claims, format_claims_for_grading


class TestExtractClaims:
    def test_absolute_bpb_claim(self):
        text = "Our method achieves 1.08 bpb on FineWeb validation set."
        claims = extract_claims(text)
        assert len(claims) >= 1
        found = [c for c in claims if "1.08" in c.magnitude]
        assert found, f"Expected claim with magnitude 1.08, got {claims}"
        assert found[0].direction in ("absolute", "improvement")

    def test_delta_percentage_claim(self):
        text = "This reduces perplexity by 3.2% on WikiText-103."
        claims = extract_claims(text)
        assert len(claims) >= 1
        found = [c for c in claims if "3.2" in c.magnitude]
        assert found, f"Expected claim with magnitude 3.2, got {claims}"

    def test_range_claim(self):
        text = "Performance improved from 1.20 to 1.08 bpb on the validation set."
        claims = extract_claims(text)
        found = [c for c in claims if c.direction == "reduction"]
        assert found, f"Expected reduction claim, got {claims}"

    def test_no_claims_in_generic_text(self):
        text = "We propose a novel architecture for language modeling."
        claims = extract_claims(text)
        assert len(claims) == 0

    def test_multiple_claims(self):
        text = (
            "Our approach achieves 1.08 bpb on FineWeb. "
            "This represents a 5% improvement over the baseline. "
            "Training time is reduced by 20% compared to prior work."
        )
        claims = extract_claims(text)
        assert len(claims) >= 2

    def test_deduplication(self):
        text = "achieves 1.08 bpb on FineWeb. achieves 1.08 bpb on FineWeb."
        claims = extract_claims(text)
        # Should deduplicate identical spans
        bpb_claims = [c for c in claims if "1.08" in c.magnitude]
        assert len(bpb_claims) <= 1

    def test_format_claims_empty(self):
        assert format_claims_for_grading([]) == ""

    def test_format_claims_nonempty(self):
        claims = [
            Claim(
                technique="QKnorm",
                metric="bpb",
                magnitude="0.05",
                direction="reduction",
                condition="on FineWeb",
                raw_text="reduces bpb by 0.05 on FineWeb",
            )
        ]
        result = format_claims_for_grading(claims)
        assert "EXTRACTED CLAIMS" in result
        assert "reduction" in result
        assert "0.05" in result


class TestExtractClaimsEdgeCases:
    def test_percentage_improvement_pattern(self):
        text = "3.5% improvement in accuracy on ImageNet."
        claims = extract_claims(text)
        found = [c for c in claims if "3.5" in c.magnitude]
        assert found

    def test_negative_sign_implies_reduction(self):
        text = "The method shows -0.03 bpb compared to baseline."
        claims = extract_claims(text)
        found = [c for c in claims if "-0.03" in c.magnitude or "0.03" in c.magnitude]
        assert found

    def test_positive_sign_implies_improvement(self):
        text = "We observe +2.1% over the previous state of the art."
        claims = extract_claims(text)
        found = [c for c in claims if "2.1" in c.magnitude]
        assert found
