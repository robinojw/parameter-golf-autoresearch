import asyncio
import hashlib
import os
from datetime import datetime, timezone

from tavily import TavilyClient

from research.fetch import RawItem

TAVILY_RELEVANCE_THRESHOLD = 0.4

BREAKING_QUERIES = [
    {
        "query": "parameter golf challenge leaderboard new record 2026",
        "time_range": "day",
        "topic": "news",
        "search_depth": "basic",
        "max_results": 5,
    },
    {
        "query": "large language model quantization breakthrough 2026",
        "time_range": "day",
        "topic": "news",
        "search_depth": "basic",
        "max_results": 5,
    },
]


def _result_to_item(result: dict) -> RawItem:
    return RawItem(
        id=f"tavily:{hashlib.md5(result['url'].encode()).hexdigest()[:12]}",
        source="tavily_breaking",
        dimension=["ml"],
        title=result["title"],
        abstract="",
        url=result["url"],
        published_date=result.get("published_date", ""),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        content_snippet=result["content"],
        raw_type="web_result",
        tavily_score=result.get("score", 0.0),
    )


async def fetch_tavily_breaking(since_hours: int = 24) -> list[RawItem]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)
    items: list[RawItem] = []

    for q in BREAKING_QUERIES:
        try:
            response = await asyncio.to_thread(
                client.search,
                query=q["query"],
                search_depth=q["search_depth"],
                time_range=q["time_range"],
                max_results=q["max_results"],
                topic=q["topic"],
                include_answer=False,
                include_raw_content=False,
            )
        except Exception:
            await asyncio.sleep(0.5)
            continue

        for result in response.get("results", []):
            score = result.get("score", 0.0)
            if score < TAVILY_RELEVANCE_THRESHOLD:
                continue
            items.append(_result_to_item(result))

        await asyncio.sleep(0.5)

    return items
