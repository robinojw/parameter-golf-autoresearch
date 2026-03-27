from datetime import datetime, timezone

import httpx

from research.fetch import RawItem

TARGET_CATEGORIES = ["language modeling", "model compression", "quantization"]
CODESOTA_BASE = "https://www.codesota.com"


async def fetch_codesota(since_hours: int = 48) -> list[RawItem]:
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        items_from_api = await _try_json_api(client, seen_ids)
        if items_from_api:
            items.extend(items_from_api)
        else:
            items_from_html = await _try_html_fallback(client, seen_ids)
            items.extend(items_from_html)

    return items


async def _try_json_api(client: httpx.AsyncClient, seen_ids: set[str]) -> list[RawItem]:
    items: list[RawItem] = []
    try:
        resp = await client.get(f"{CODESOTA_BASE}/api/benchmarks")
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return items

    benchmarks = (
        data
        if isinstance(data, list)
        else data.get("benchmarks", data.get("results", []))
    )
    if not isinstance(benchmarks, list):
        return items

    for entry in benchmarks:
        category = (entry.get("category", "") or entry.get("task", "")).lower()
        if not any(target in category for target in TARGET_CATEGORIES):
            continue

        entry_id = str(entry.get("id", "") or entry.get("name", ""))
        if not entry_id or entry_id in seen_ids:
            continue

        title = entry.get("name", "") or entry.get("title", "")
        description = entry.get("description", "") or entry.get("abstract", "")
        url = entry.get("url", "") or f"{CODESOTA_BASE}/benchmark/{entry_id}"
        metric = entry.get("metric", "")
        best_value = entry.get("best_value", "") or entry.get("sota_value", "")

        snippet = f"Metric: {metric}, Best: {best_value}" if metric else ""

        seen_ids.add(entry_id)
        items.append(
            RawItem(
                id=f"codesota:{entry_id}",
                source="codesota",
                dimension=["ml"],
                title=title,
                abstract=description,
                url=url,
                published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                content_snippet=snippet,
                raw_type="benchmark_entry",
            )
        )

    return items


async def _try_html_fallback(
    client: httpx.AsyncClient, seen_ids: set[str]
) -> list[RawItem]:
    items: list[RawItem] = []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return items

    for category in TARGET_CATEGORIES:
        slug = category.replace(" ", "-")
        try:
            resp = await client.get(f"{CODESOTA_BASE}/category/{slug}")
            resp.raise_for_status()
        except httpx.HTTPError:
            try:
                resp = await client.get(f"{CODESOTA_BASE}/area/{slug}")
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for row in soup.select(
            "table tr, .benchmark-card, .result-row, [data-benchmark]"
        ):
            link = row.find("a")
            if not link:
                continue

            title = link.get_text(strip=True)
            href = link.get("href", "")
            if not title:
                continue

            entry_id = (
                href.strip("/").split("/")[-1]
                if href
                else title.lower().replace(" ", "-")
            )
            if entry_id in seen_ids:
                continue

            full_url = href if href.startswith("http") else f"{CODESOTA_BASE}{href}"

            cells = row.find_all("td") if row.name == "tr" else []
            snippet = (
                " | ".join(c.get_text(strip=True) for c in cells[1:4])
                if len(cells) > 1
                else ""
            )

            seen_ids.add(entry_id)
            items.append(
                RawItem(
                    id=f"codesota:{entry_id}",
                    source="codesota",
                    dimension=["ml"],
                    title=title,
                    abstract=f"Benchmark entry from CodeSOTA: {category}",
                    url=full_url,
                    published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    content_snippet=snippet,
                    raw_type="benchmark_entry",
                )
            )

    return items
