"""Tests for source authority tiers and fail-open grading thresholds."""

from research.fetch import RawItem, get_authority_tier, AUTHORITY_RELEVANCE_FLOORS
from research.grade import _score_to_tier


class TestAuthorityTiers:
    def _make_item(self, source: str = "", url: str = "") -> RawItem:
        return RawItem(
            id="test:1",
            source=source,
            dimension=["ml"],
            title="Test",
            abstract="Test abstract",
            url=url,
            published_date="2026-01-01",
        )

    def test_arxiv_source_is_tier_1(self):
        item = self._make_item(source="arxiv", url="https://arxiv.org/abs/2401.12345")
        assert get_authority_tier(item) == 1

    def test_semantic_scholar_source_is_tier_1(self):
        item = self._make_item(source="semantic_scholar", url="https://semanticscholar.org/paper/abc")
        assert get_authority_tier(item) == 1

    def test_openreview_source_is_tier_1(self):
        item = self._make_item(source="openreview", url="https://openreview.net/forum?id=abc")
        assert get_authority_tier(item) == 1

    def test_github_prs_is_tier_2(self):
        item = self._make_item(source="github_prs", url="https://github.com/openai/parameter-golf/pull/1")
        assert get_authority_tier(item) == 2

    def test_github_code_search_is_tier_2(self):
        item = self._make_item(source="github_code_search", url="https://github.com/search?q=muon")
        assert get_authority_tier(item) == 2

    def test_unknown_source_with_arxiv_url_is_tier_1(self):
        item = self._make_item(source="tavily_scheduled", url="https://arxiv.org/abs/2401.99999")
        assert get_authority_tier(item) == 1

    def test_unknown_source_with_github_url_is_tier_2(self):
        item = self._make_item(source="tavily_scheduled", url="https://github.com/karpathy/autoresearch")
        assert get_authority_tier(item) == 2

    def test_generic_blog_is_tier_3(self):
        item = self._make_item(source="tavily_scheduled", url="https://blog.example.com/llm-tips")
        assert get_authority_tier(item) == 3

    def test_tier_1_has_lowest_relevance_floor(self):
        assert AUTHORITY_RELEVANCE_FLOORS[1] < AUTHORITY_RELEVANCE_FLOORS[2]
        assert AUTHORITY_RELEVANCE_FLOORS[2] < AUTHORITY_RELEVANCE_FLOORS[3]

    def test_icml_proceedings_is_tier_1(self):
        item = self._make_item(source="", url="https://proceedings.mlr.press/v202/paper.html")
        assert get_authority_tier(item) == 1

    def test_huggingface_is_tier_2(self):
        item = self._make_item(source="", url="https://huggingface.co/papers/2401.12345")
        assert get_authority_tier(item) == 2


class TestFailOpenGrading:
    def test_normal_tier_a_threshold(self):
        assert _score_to_tier(12.0, has_competitors=False, fail_open=False) == "A"
        assert _score_to_tier(11.9, has_competitors=False, fail_open=False) == "B"

    def test_fail_open_raises_tier_a_threshold(self):
        # Normal: 12 is Tier A. Fail-open: needs 14 for Tier A
        assert _score_to_tier(12.0, has_competitors=False, fail_open=True) == "B"
        assert _score_to_tier(13.9, has_competitors=False, fail_open=True) == "B"
        assert _score_to_tier(14.0, has_competitors=False, fail_open=True) == "A"

    def test_fail_open_raises_tier_b_threshold(self):
        # Normal: 8 is Tier B. Fail-open: needs 9 for Tier B
        assert _score_to_tier(8.0, has_competitors=False, fail_open=True) == "C"
        assert _score_to_tier(9.0, has_competitors=False, fail_open=True) == "B"

    def test_fail_open_with_competitors(self):
        # With competitors: normal Tier A = 14, fail-open = 16
        assert _score_to_tier(14.0, has_competitors=True, fail_open=True) == "B"
        assert _score_to_tier(16.0, has_competitors=True, fail_open=True) == "A"

    def test_no_penalty_without_fail_open(self):
        assert _score_to_tier(12.0, has_competitors=False, fail_open=False) == "A"
        assert _score_to_tier(8.0, has_competitors=False, fail_open=False) == "B"
