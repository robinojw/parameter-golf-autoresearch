import asyncio
import hashlib
import os
from datetime import datetime, timezone

import httpx

from research.fetch import RawItem

GITHUB_API = "https://api.github.com"
_SEARCH_ENDPOINT = "/search/code"
_RATE_LIMIT_SLEEP = 2.0
_REQUEST_TIMEOUT = 30.0
_MAX_RESULTS_PER_QUERY = 5
_SNIPPET_MAX_LEN = 1500
_HASH_TRUNCATE = 12
_DIM_ML = "ml"
_DIM_INFRA = "infra"
_DIM_KEY = "dimension"
_Q_KEY = "q"
_SOURCE_NAME = "github_code_search"
_RAW_TYPE = "code_snippet"

CODE_SEARCH_QUERIES = [
    {
        _Q_KEY: "quantization aware training language:python path:train",
        _DIM_KEY: _DIM_ML,
    },
    {
        _Q_KEY: "muon optimizer language:python",
        _DIM_KEY: _DIM_ML,
    },
    {
        _Q_KEY: "ternary quantization weight language:python",
        _DIM_KEY: _DIM_ML,
    },
    {
        _Q_KEY: "bits per byte fineweb language:python",
        _DIM_KEY: _DIM_ML,
    },
    {
        _Q_KEY: "BigramHash embedding language:python",
        _DIM_KEY: _DIM_ML,
    },
    {
        _Q_KEY: "triton kernel quantized matmul language:python",
        _DIM_KEY: _DIM_INFRA,
    },
    {
        _Q_KEY: "zstd compress weights language:python",
        _DIM_KEY: _DIM_INFRA,
    },
    {
        _Q_KEY: "parameter golf train_gpt language:python",
        _DIM_KEY: _DIM_ML,
    },
]


def _build_item_id(html_url: str) -> str:
    url_hash = hashlib.md5(html_url.encode()).hexdigest()[:_HASH_TRUNCATE]
    return f"ghcode:{url_hash}"


def _extract_snippet(item: dict) -> str:
    fragments = item.get("text_matches", [])
    if not fragments:
        return ""
    parts: list[str] = []
    for match in fragments:
        fragment_text = match.get("fragment", "")
        if fragment_text:
            parts.append(fragment_text)
    return "\n---\n".join(parts)[:_SNIPPET_MAX_LEN]


async def fetch_github_code_search() -> list[RawItem]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.text-match+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    items: list[RawItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT, headers=headers) as client:
        for query_spec in CODE_SEARCH_QUERIES:
            try:
                resp = await client.get(
                    f"{GITHUB_API}{_SEARCH_ENDPOINT}",
                    params={
                        _Q_KEY: query_spec[_Q_KEY],
                        "per_page": _MAX_RESULTS_PER_QUERY,
                        "sort": "indexed",
                        "order": "desc",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError):
                continue

            for result in data.get("items", []):
                html_url = result.get("html_url", "")
                if not html_url:
                    continue

                item_id = _build_item_id(html_url)
                if item_id in seen_ids:
                    continue

                repo_info = result.get("repository", {})
                repo_name = repo_info.get("full_name", "")
                file_path = result.get("path", "")

                title = f"[{repo_name}] {file_path}"
                snippet = _extract_snippet(result)
                abstract = (
                    f"Code match in {repo_name}/{file_path} "
                    f"for query: {query_spec[_Q_KEY]}"
                )

                seen_ids.add(item_id)
                items.append(
                    RawItem(
                        id=item_id,
                        source=_SOURCE_NAME,
                        dimension=[query_spec[_DIM_KEY]],
                        title=title,
                        abstract=abstract,
                        url=html_url,
                        published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        content_snippet=snippet,
                        raw_type=_RAW_TYPE,
                    )
                )

            await asyncio.sleep(_RATE_LIMIT_SLEEP)

    return items
