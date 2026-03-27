import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from research.fetch import RawItem

VENUES = [
    "ICLR.cc/2026/Conference",
    "NeurIPS.cc/2025/Conference",
    "ICML.cc/2025/Conference",
    "MLSys.org/2026/Conference",
]
KEYWORDS = [
    "quantization",
    "compression",
    "parameter efficient",
    "weight sharing",
    "test-time training",
    "bits per byte",
    "low-rank",
    "tokenizer",
    "optimizer",
    "state space",
]
RATE_LIMIT_SECONDS = 1
MAX_PER_VENUE = 100


async def fetch_openreview(since_hours: int = 48) -> list[RawItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    cutoff_ts_ms = int(cutoff.timestamp() * 1000)
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        request_count = 0
        for venue in VENUES:
            for keyword in KEYWORDS:
                if request_count > 0:
                    await asyncio.sleep(RATE_LIMIT_SECONDS)
                request_count += 1

                params = {
                    "content.venue": venue,
                    "content.keywords": keyword,
                    "limit": MAX_PER_VENUE,
                    "sort": "cdate:desc",
                }

                try:
                    resp = await client.get(
                        "https://api2.openreview.net/notes", params=params
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except (httpx.HTTPError, ValueError):
                    continue

                notes = data.get("notes", [])
                for note in notes:
                    note_id = note.get("id", "")
                    if not note_id or note_id in seen_ids:
                        continue

                    cdate = note.get("cdate", 0)
                    if cdate < cutoff_ts_ms:
                        continue

                    content = note.get("content", {})

                    title_val = content.get("title", {})
                    title = (
                        title_val.get("value", "")
                        if isinstance(title_val, dict)
                        else str(title_val)
                    )

                    abstract_val = content.get("abstract", {})
                    abstract = (
                        abstract_val.get("value", "")
                        if isinstance(abstract_val, dict)
                        else str(abstract_val)
                    )

                    if not title:
                        continue

                    pub_date = datetime.fromtimestamp(
                        cdate / 1000, tz=timezone.utc
                    ).isoformat()

                    forum_id = note.get("forum", note_id)
                    url = f"https://openreview.net/forum?id={forum_id}"

                    seen_ids.add(note_id)
                    items.append(
                        RawItem(
                            id=f"openreview:{note_id}",
                            source="openreview",
                            dimension=["ml"],
                            title=title,
                            abstract=abstract,
                            url=url,
                            published_date=pub_date,
                            raw_type="paper",
                        )
                    )

                if len(items) >= MAX_PER_VENUE * len(VENUES):
                    break
            if len(items) >= MAX_PER_VENUE * len(VENUES):
                break

    return items
