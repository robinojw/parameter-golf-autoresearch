import json
import os
from pathlib import Path

from tavily import TavilyClient

from research.fetch import RawItem

EXTRACT_CACHE_PATH = Path("extract_cache.jsonl")


def _load_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if not EXTRACT_CACHE_PATH.exists():
        return cache
    with EXTRACT_CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["url"]] = entry["content"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _save_to_cache(url: str, content: str) -> None:
    with EXTRACT_CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "content": content}) + "\n")


async def extract_url(url: str) -> str | None:
    cache = _load_cache()
    if url in cache:
        return cache[url]

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    client = TavilyClient(api_key=api_key)
    try:
        response = client.extract(urls=[url])
    except Exception:
        return None

    results = response.get("results", [])
    if not results:
        return None

    content = results[0].get("raw_content", "")
    if not content:
        return None

    _save_to_cache(url, content)
    return content


async def extract_if_needed(
    item: RawItem,
    graded_score: float,
    score_threshold: float = 10.0,
    snippet_min_length: int = 300,
) -> str | None:
    if graded_score < score_threshold:
        return None
    if len(item.content_snippet or "") >= snippet_min_length:
        return None
    return await extract_url(item.url)
