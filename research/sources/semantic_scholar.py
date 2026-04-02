import asyncio
import os
from datetime import datetime, timedelta, timezone

import httpx

from research.fetch import RawItem

SEMANTIC_QUERIES = [
    "quantization aware training language model compression",
    "muon optimizer second order neural network",
    "ternary binary weight language model perplexity",
    "information theoretic compression neural network",
    "test time training inference compute language model",
]
CITATION_LOOKBACK_DAYS = 90

API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,url,publicationDate,citationCount"
RATE_LIMIT_SECONDS = 1
DEFAULT_SINCE_HOURS = 48
RESULTS_PER_QUERY = 20
HTTP_TIMEOUT = 30.0
DATE_FORMAT = "%Y-%m-%d"

MATH_KEYWORDS = ["entropy", "information theory", "coding theory"]
INFRA_KEYWORDS = ["gpu", "kernel", "hardware", "accelerator"]
_KEY_PAPER_ID = "paperId"


def _is_too_old(pub_date_str: str) -> bool:
    if not pub_date_str:
        return False
    try:
        pub_date = datetime.strptime(pub_date_str, DATE_FORMAT).replace(
            tzinfo=timezone.utc
        )
        lookback_cutoff = datetime.now(timezone.utc) - timedelta(
            days=CITATION_LOOKBACK_DAYS
        )
        return pub_date < lookback_cutoff
    except ValueError:
        return False


def _infer_dimensions(title: str, abstract: str) -> list[str]:
    combined = (title + " " + abstract).lower()
    dimensions = ["ml"]
    if any(kw in combined for kw in MATH_KEYWORDS):
        dimensions.append("math")
    if any(kw in combined for kw in INFRA_KEYWORDS):
        dimensions.append("infra")
    return dimensions


def _paper_to_item(paper: dict) -> RawItem | None:
    paper_id = paper.get(_KEY_PAPER_ID, "")
    title = paper.get("title", "")
    has_required_fields = paper_id and title
    if not has_required_fields:
        return None

    abstract = paper.get("abstract", "") or ""
    url = paper.get("url", "") or f"https://www.semanticscholar.org/paper/{paper_id}"
    pub_date_str = paper.get("publicationDate", "")

    if _is_too_old(pub_date_str):
        return None

    return RawItem(
        id=f"s2:{paper_id}",
        source="semantic_scholar",
        dimension=_infer_dimensions(title, abstract),
        title=title,
        abstract=abstract,
        url=url,
        published_date=pub_date_str or datetime.now(timezone.utc).strftime(DATE_FORMAT),
        content_snippet=f"Citations: {paper.get('citationCount', 0)}",
        raw_type="paper",
    )


async def fetch_semantic_scholar(
    since_hours: int = DEFAULT_SINCE_HOURS,
) -> list[RawItem]:
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    api_key = os.environ.get("S2_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        for i, query in enumerate(SEMANTIC_QUERIES):
            if i > 0:
                await asyncio.sleep(RATE_LIMIT_SECONDS)

            params = {
                "query": query,
                "limit": RESULTS_PER_QUERY,
                "fields": FIELDS,
            }

            try:
                resp = await client.get(f"{API_BASE}/paper/search", params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError):
                continue

            for paper in data.get("data", []):
                item = _paper_to_item(paper)
                if item is None:
                    continue
                raw_id = paper.get(_KEY_PAPER_ID, "")
                if raw_id in seen_ids:
                    continue
                seen_ids.add(raw_id)
                items.append(item)

    return items
