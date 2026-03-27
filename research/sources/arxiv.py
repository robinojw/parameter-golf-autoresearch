import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import httpx

from research.fetch import RawItem

ARXIV_QUERIES = [
    ("cs.LG", "abs:quantization AND abs:language model"),
    ("cs.LG", "abs:parameter efficient AND abs:compression"),
    ("cs.LG", "abs:test time training AND abs:language model"),
    ("cs.LG", "abs:bits per byte"),
    ("cs.LG", "abs:weight tying AND abs:transformer"),
    ("cs.IT", "abs:neural network AND abs:entropy coding"),
    ("math.OC", "abs:optimizer AND abs:neural network convergence"),
    ("cs.AR", "abs:quantization AND abs:GPU AND abs:kernel"),
]
DATE_WINDOW_DAYS = 14
MAX_RESULTS_PER_QUERY = 20
RATE_LIMIT_SECONDS = 3

DIMENSION_MAP = {
    "cs.LG": "ml",
    "cs.AI": "ml",
    "cs.CL": "ml",
    "cs.IT": "math",
    "math.OC": "math",
    "math.IT": "math",
    "cs.AR": "infra",
    "cs.DC": "infra",
    "cs.PF": "infra",
}

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


async def fetch_arxiv(since_hours: int = 48) -> list[RawItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, (category, query) in enumerate(ARXIV_QUERIES):
            if i > 0:
                await asyncio.sleep(RATE_LIMIT_SECONDS)

            search_query = f"cat:{category} AND {query}"
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": MAX_RESULTS_PER_QUERY,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            try:
                resp = await client.get(
                    "https://export.arxiv.org/api/query", params=params
                )
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            root = ElementTree.fromstring(resp.text)

            for entry in root.findall(f"{ATOM_NS}entry"):
                arxiv_id_el = entry.find(f"{ATOM_NS}id")
                title_el = entry.find(f"{ATOM_NS}title")
                summary_el = entry.find(f"{ATOM_NS}summary")
                published_el = entry.find(f"{ATOM_NS}published")
                updated_el = entry.find(f"{ATOM_NS}updated")

                if (
                    arxiv_id_el is None
                    or title_el is None
                    or summary_el is None
                    or published_el is None
                ):
                    continue

                arxiv_url = arxiv_id_el.text.strip()
                arxiv_id = (
                    arxiv_url.split("/abs/")[-1]
                    if "/abs/" in arxiv_url
                    else arxiv_url.split("/")[-1]
                )

                if arxiv_id in seen_ids:
                    continue

                pub_date_str = (
                    updated_el.text if updated_el is not None else published_el.text
                ).strip()
                try:
                    pub_date = datetime.fromisoformat(
                        pub_date_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue

                date_window_cutoff = datetime.now(timezone.utc) - timedelta(
                    days=DATE_WINDOW_DAYS
                )
                if pub_date < date_window_cutoff:
                    continue

                categories = entry.findall(
                    f"{ARXIV_NS}primary_category"
                ) + entry.findall(f"{ATOM_NS}category")
                dimensions: list[str] = []
                for cat_el in categories:
                    term = cat_el.get("term", "")
                    if term in DIMENSION_MAP and DIMENSION_MAP[term] not in dimensions:
                        dimensions.append(DIMENSION_MAP[term])

                if not dimensions:
                    dim = DIMENSION_MAP.get(category, "ml")
                    dimensions = [dim]

                seen_ids.add(arxiv_id)
                items.append(
                    RawItem(
                        id=f"arxiv:{arxiv_id}",
                        source="arxiv",
                        dimension=dimensions,
                        title=title_el.text.strip().replace("\n", " "),
                        abstract=summary_el.text.strip().replace("\n", " "),
                        url=arxiv_url,
                        published_date=pub_date_str,
                        raw_type="paper",
                    )
                )

    return items
