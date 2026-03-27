import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from research.fetch import RawItem

FEEDS = [
    ("https://developer.nvidia.com/blog/feed/", "infra"),
    ("https://mlsys.org/Conferences/2026/rss.xml", "infra"),
    ("https://lilianweng.github.io/index.xml", "ml"),
    ("https://magazine.sebastianraschka.com/feed", "ml"),
]


async def fetch_feeds(since_hours: int = 48) -> list[RawItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for feed_url, dimension in FEEDS:
            try:
                resp = await client.get(feed_url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            feed = feedparser.parse(resp.text)

            for entry in feed.entries:
                pub_date = _parse_entry_date(entry)
                if pub_date and pub_date < cutoff:
                    continue

                link = entry.get("link", "")
                title = entry.get("title", "")
                if not title:
                    continue

                entry_id = (
                    entry.get("id", link)
                    or hashlib.sha256((title + link).encode()).hexdigest()[:16]
                )

                if entry_id in seen_ids:
                    continue

                summary = entry.get("summary", "") or entry.get("description", "")
                content_list = entry.get("content", [])
                content_snippet = (
                    content_list[0].get("value", "")[:500] if content_list else ""
                )

                pub_date_str = (
                    pub_date.isoformat()
                    if pub_date
                    else datetime.now(timezone.utc).strftime("%Y-%m-%d")
                )

                seen_ids.add(entry_id)
                items.append(
                    RawItem(
                        id=f"feed:{hashlib.sha256(entry_id.encode()).hexdigest()[:16]}",
                        source=f"feed:{feed_url.split('/')[2]}",
                        dimension=[dimension],
                        title=title,
                        abstract=summary[:1000],
                        url=link,
                        published_date=pub_date_str,
                        content_snippet=content_snippet,
                        raw_type="blog_post",
                    )
                )

    return items


def _parse_entry_date(entry: dict) -> datetime | None:
    for date_field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(date_field)
        if parsed:
            try:
                from time import mktime

                return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                continue

    for date_field in ("published", "updated"):
        date_str = entry.get(date_field, "")
        if date_str:
            try:
                return parsedate_to_datetime(date_str).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

    return None
