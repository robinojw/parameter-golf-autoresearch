import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

RAW_CACHE_PATH = Path("raw_cache.jsonl")
TAVILY_RELEVANCE_THRESHOLD = 0.4


@dataclass
class RawItem:
    id: str
    source: str
    dimension: list[str]
    title: str
    abstract: str
    url: str
    published_date: str
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    content_snippet: str = ""
    raw_type: str = "paper"
    tavily_score: float = 0.0


@dataclass
class GradedItem:
    id: str
    score: float
    tier: str
    score_breakdown: dict
    agent_summary: str
    flags: list[str]
    graded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    grade_error: bool = False


async def fetch_fast(since_hours: int = 48) -> list[RawItem]:
    """Fetch from fast-moving sources: GitHub PRs, GitHub code search, Tavily scheduled.

    These sources update continuously — competitors merge PRs at any time,
    web content appears hourly. Safe to call every 1-2 hours.
    """
    from research.sources.github_prs import fetch_github_prs
    from research.sources.github_code_search import fetch_github_code_search
    from research.sources.tavily_scheduled import fetch_tavily_scheduled

    results = await asyncio.gather(
        fetch_github_prs(since_hours),
        fetch_github_code_search(),
        fetch_tavily_scheduled(),
        return_exceptions=True,
    )

    return _dedup_and_cache(results)


async def fetch_slow(since_hours: int = 48) -> list[RawItem]:
    """Fetch from slow-moving sources: ArXiv, OpenReview, Semantic Scholar, RSS, CodeSOTA.

    These sources update daily at most. No benefit to checking more than every 12-24h.
    """
    from research.sources.arxiv import fetch_arxiv
    from research.sources.openreview import fetch_openreview
    from research.sources.semantic_scholar import fetch_semantic_scholar
    from research.sources.feeds import fetch_feeds
    from research.sources.codesota import fetch_codesota

    results = await asyncio.gather(
        fetch_arxiv(since_hours),
        fetch_openreview(since_hours),
        fetch_semantic_scholar(since_hours),
        fetch_feeds(since_hours),
        fetch_codesota(),
        return_exceptions=True,
    )

    return _dedup_and_cache(results)


async def fetch_all(since_hours: int = 48) -> list[RawItem]:
    """Fetch from all sources. Used for full refresh cycles."""
    from research.sources.arxiv import fetch_arxiv
    from research.sources.openreview import fetch_openreview
    from research.sources.semantic_scholar import fetch_semantic_scholar
    from research.sources.github_prs import fetch_github_prs
    from research.sources.feeds import fetch_feeds
    from research.sources.tavily_scheduled import fetch_tavily_scheduled
    from research.sources.github_code_search import fetch_github_code_search

    results = await asyncio.gather(
        fetch_arxiv(since_hours),
        fetch_openreview(since_hours),
        fetch_semantic_scholar(since_hours),
        fetch_github_prs(since_hours),
        fetch_feeds(since_hours),
        fetch_tavily_scheduled(),
        fetch_github_code_search(),
        return_exceptions=True,
    )

    return _dedup_and_cache(results)


def _dedup_and_cache(results: list) -> list[RawItem]:
    """Deduplicate fetch results against the cache and append new items."""
    existing_ids = _load_existing_ids()
    all_items: list[RawItem] = []
    seen_ids: set[str] = set()

    for result in results:
        if isinstance(result, Exception):
            print(f"[fetch] source failed: {result}")
            continue
        for item in result:
            is_new = item.id not in existing_ids and item.id not in seen_ids
            if not is_new:
                continue
            below_tavily_threshold = (
                item.tavily_score > 0 and item.tavily_score < TAVILY_RELEVANCE_THRESHOLD
            )
            if below_tavily_threshold:
                continue
            seen_ids.add(item.id)
            all_items.append(item)

    _append_to_cache(all_items)
    return all_items


def _load_existing_ids() -> set[str]:
    ids: set[str] = set()
    if not RAW_CACHE_PATH.exists():
        return ids
    with open(RAW_CACHE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _append_to_cache(items: list[RawItem]) -> None:
    if not items:
        return
    with open(RAW_CACHE_PATH, "a") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
