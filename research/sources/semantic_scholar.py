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

# Forward citation check fields
CITATION_FIELDS = "title,abstract,url,publicationDate,citationCount"
MAX_FORWARD_CITATIONS = 10


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


async def get_forward_citations(paper_id: str) -> list[dict]:
    """Fetch papers that cite the given paper (forward citations).

    Uses the Semantic Scholar API citations endpoint. Returns a list of
    dicts with keys: title, abstract, url, publicationDate, citationCount.

    This is a structured, deterministic corroboration source — not circular
    with web search. If a paper's claims are cited by subsequent work,
    that's meaningful quality signal.

    Args:
        paper_id: Semantic Scholar paper ID (e.g. "abc123" from "s2:abc123")

    Returns:
        List of citing paper dicts, sorted by citation count descending.
        Empty list on error or no citations.
    """
    api_key = os.environ.get("S2_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.get(
                f"{API_BASE}/paper/{paper_id}/citations",
                params={
                    "fields": CITATION_FIELDS,
                    "limit": MAX_FORWARD_CITATIONS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            print(f"[semantic_scholar] forward citation lookup failed for {paper_id}: {exc}")
            return []

    citations = []
    for entry in data.get("data", []):
        citing_paper = entry.get("citingPaper", {})
        if not citing_paper.get("title"):
            continue
        citations.append({
            "title": citing_paper.get("title", ""),
            "abstract": (citing_paper.get("abstract") or "")[:500],
            "url": citing_paper.get("url", ""),
            "publicationDate": citing_paper.get("publicationDate", ""),
            "citationCount": citing_paper.get("citationCount", 0),
        })

    # Sort by citation count descending (most-cited first)
    citations.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
    return citations


def format_citation_evidence(citations: list[dict]) -> str:
    """Format forward citations as a markdown evidence block for verification.

    Returns empty string if no citations.
    """
    if not citations:
        return ""
    lines = [f"**Forward citations ({len(citations)} papers cite this work):**"]
    for i, c in enumerate(citations[:5], 1):
        cite_count = c.get("citationCount", 0)
        lines.append(
            f"  {i}. [{c['title']}]({c.get('url', '')}) "
            f"— {cite_count} citations, {c.get('publicationDate', 'n/a')}"
        )
        abstract = c.get("abstract", "")
        if abstract:
            lines.append(f"     > {abstract[:200]}")
    return "\n".join(lines)
