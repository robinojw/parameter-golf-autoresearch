import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tavily import TavilyClient

from research.fetch import RawItem


def _result_to_item(result: dict) -> RawItem:
    return RawItem(
        id=f"tavily:{hashlib.md5(result['url'].encode()).hexdigest()[:12]}",
        source="tavily_agent",
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


def _append_to_raw_cache(items: list[RawItem]) -> None:
    cache_path = Path("raw_cache.jsonl")
    with cache_path.open("a", encoding="utf-8") as f:
        for item in items:
            row = {
                "id": item.id,
                "source": item.source,
                "dimension": item.dimension,
                "title": item.title,
                "abstract": item.abstract,
                "url": item.url,
                "published_date": item.published_date,
                "fetched_at": item.fetched_at,
                "content_snippet": item.content_snippet,
                "raw_type": item.raw_type,
                "tavily_score": item.tavily_score,
            }
            f.write(json.dumps(row) + "\n")


def _format_markdown(query: str, results: list[dict], depth: str) -> str:
    lines = [
        f'## Search Results: "{query}"',
        f"*Tavily search — {len(results)} results, {depth} depth*",
        "",
    ]
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        lines.append(f"### {i}. [{r['title']}]({r['url']}) — relevance: {score:.2f}")
        lines.append(f"> {r['content']}")
        lines.append("")
    return "\n".join(lines)


def agent_search(query: str, depth: str = "advanced", max_results: int = 5) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set"

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth=depth,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = response.get("results", [])
    items = [_result_to_item(r) for r in results]
    _append_to_raw_cache(items)

    return _format_markdown(query, results, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--depth", default="advanced", choices=["basic", "advanced"])
    parser.add_argument("--max-results", default=5, type=int)
    args = parser.parse_args()

    if not os.environ.get("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    output = agent_search(args.query, args.depth, args.max_results)
    print(output)
