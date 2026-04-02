import asyncio
import hashlib
import os
from datetime import datetime, timezone

from tavily import TavilyClient

from research.fetch import RawItem, TAVILY_RELEVANCE_THRESHOLD

SCHEDULED_QUERIES = [
    {
        "query": "1-bit ternary quantization language model bits per byte 2026",
        "search_depth": "advanced",
        "time_range": "month",
        "max_results": 8,
        "dimension": "ml",
        "include_domains": [
            "arxiv.org",
            "huggingface.co",
            "github.com",
            "openreview.net",
            "proceedings.mlr.press",
        ],
    },
    {
        "query": "test time training inference compute language model compression 2026",
        "search_depth": "advanced",
        "time_range": "month",
        "max_results": 8,
        "dimension": "ml",
    },
    {
        "query": "parameter efficient transformer weight tying bits per byte FineWeb",
        "search_depth": "basic",
        "time_range": "month",
        "max_results": 6,
        "dimension": "ml",
    },
    {
        "query": "openai parameter golf challenge leaderboard technique 2026",
        "search_depth": "advanced",
        "time_range": "week",
        "max_results": 10,
        "dimension": "ml",
    },
    {
        "query": "modded nanogpt speedrun new optimizer trick 2026",
        "search_depth": "basic",
        "time_range": "week",
        "max_results": 6,
        "dimension": "ml",
    },
    {
        "query": "neural network quantization information theory rate distortion",
        "search_depth": "basic",
        "time_range": "month",
        "max_results": 5,
        "dimension": "math",
        "include_domains": ["arxiv.org", "jmlr.org", "proceedings.mlr.press"],
    },
    {
        "query": "Muon optimizer second order convergence analysis",
        "search_depth": "basic",
        "time_range": "month",
        "max_results": 5,
        "dimension": "math",
    },
    {
        "query": "CUDA triton kernel int4 int6 quantization matmul",
        "search_depth": "advanced",
        "time_range": "month",
        "max_results": 6,
        "dimension": "infra",
        "include_domains": [
            "github.com",
            "developer.nvidia.com",
            "triton-lang.org",
            "pytorch.org",
        ],
    },
    {
        "query": "zstd brotli model weight compression benchmark",
        "search_depth": "basic",
        "time_range": "month",
        "max_results": 5,
        "dimension": "infra",
    },
]


def _result_to_item(result: dict, dimension: str) -> RawItem:
    return RawItem(
        id=f"tavily:{hashlib.md5(result['url'].encode()).hexdigest()[:12]}",
        source="tavily_scheduled",
        dimension=[dimension],
        title=result["title"],
        abstract="",
        url=result["url"],
        published_date=result.get("published_date", ""),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        content_snippet=result["content"],
        raw_type="web_result",
        tavily_score=result.get("score", 0.0),
    )


async def fetch_tavily_scheduled() -> list[RawItem]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)
    items: list[RawItem] = []

    for q in SCHEDULED_QUERIES:
        search_kwargs: dict = {
            "query": q["query"],
            "search_depth": q["search_depth"],
            "time_range": q["time_range"],
            "max_results": q["max_results"],
            "include_answer": False,
            "include_raw_content": False,
        }
        if "include_domains" in q:
            search_kwargs["include_domains"] = q["include_domains"]

        try:
            response = await asyncio.to_thread(client.search, **search_kwargs)
        except Exception:
            await asyncio.sleep(0.5)
            continue

        for result in response.get("results", []):
            score = result.get("score", 0.0)
            if score < TAVILY_RELEVANCE_THRESHOLD:
                continue
            items.append(_result_to_item(result, q["dimension"]))

        await asyncio.sleep(0.5)

    return items
