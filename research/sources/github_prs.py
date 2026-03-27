import os
from datetime import datetime, timedelta, timezone

import httpx

from research.fetch import RawItem

WATCHED_REPOS = [
    ("openai", "parameter-golf"),
    ("KellerJordan", "modded-nanogpt"),
    ("karpathy", "autoresearch"),
]

GITHUB_API = "https://api.github.com"


async def fetch_github_prs(since_hours: int = 48) -> list[RawItem]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    cutoff_iso = cutoff.isoformat()
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for owner, repo in WATCHED_REPOS:
            pr_items = await _fetch_prs(client, owner, repo, cutoff, seen_ids)
            items.extend(pr_items)

            if owner == "openai" and repo == "parameter-golf":
                for pr_item in pr_items:
                    record_content = await _fetch_pr_records(
                        client, owner, repo, pr_item
                    )
                    if record_content:
                        pr_item.content_snippet = record_content

            commit_items = await _fetch_commits(
                client, owner, repo, cutoff_iso, seen_ids
            )
            items.extend(commit_items)

    return items


async def _fetch_prs(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    cutoff: datetime,
    seen_ids: set[str],
) -> list[RawItem]:
    items: list[RawItem] = []
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
            params={
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "per_page": 30,
            },
        )
        resp.raise_for_status()
        pulls = resp.json()
    except (httpx.HTTPError, ValueError):
        return items

    for pr in pulls:
        updated_str = pr.get("updated_at", "")
        if updated_str:
            try:
                updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                if updated < cutoff:
                    continue
            except ValueError:
                pass

        pr_number = pr.get("number", 0)
        pr_id = f"gh:{owner}/{repo}/pr/{pr_number}"
        if pr_id in seen_ids:
            continue

        title = pr.get("title", "")
        body = pr.get("body", "") or ""
        url = pr.get("html_url", f"https://github.com/{owner}/{repo}/pull/{pr_number}")
        created_at = pr.get("created_at", "")

        dimensions = _infer_dimensions(title + " " + body)

        seen_ids.add(pr_id)
        items.append(
            RawItem(
                id=pr_id,
                source="github",
                dimension=dimensions,
                title=f"[{owner}/{repo}] PR #{pr_number}: {title}",
                abstract=body[:1000],
                url=url,
                published_date=created_at,
                raw_type="pr",
            )
        )

    return items


async def _fetch_commits(
    client: httpx.AsyncClient, owner: str, repo: str, since_iso: str, seen_ids: set[str]
) -> list[RawItem]:
    items: list[RawItem] = []
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"since": since_iso, "per_page": 20},
        )
        resp.raise_for_status()
        commits = resp.json()
    except (httpx.HTTPError, ValueError):
        return items

    for commit in commits:
        sha = commit.get("sha", "")
        if not sha:
            continue

        commit_id = f"gh:{owner}/{repo}/commit/{sha[:12]}"
        if commit_id in seen_ids:
            continue

        commit_data = commit.get("commit", {})
        message = commit_data.get("message", "")
        author_data = commit_data.get("author", {})
        date_str = author_data.get("date", "")
        url = commit.get("html_url", f"https://github.com/{owner}/{repo}/commit/{sha}")

        first_line = message.split("\n")[0]
        dimensions = _infer_dimensions(message)

        seen_ids.add(commit_id)
        items.append(
            RawItem(
                id=commit_id,
                source="github",
                dimension=dimensions,
                title=f"[{owner}/{repo}] {first_line}",
                abstract=message[:1000],
                url=url,
                published_date=date_str,
                raw_type="commit",
            )
        )

    return items


async def _fetch_pr_records(
    client: httpx.AsyncClient, owner: str, repo: str, pr_item: RawItem
) -> str:
    pr_number = pr_item.id.split("/pr/")[-1]
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files",
            params={"per_page": 50},
        )
        resp.raise_for_status()
        files = resp.json()
    except (httpx.HTTPError, ValueError):
        return ""

    for f in files:
        filename = f.get("filename", "")
        if filename.startswith("records/") and filename.endswith("README.md"):
            raw_url = f.get("raw_url", "")
            if raw_url:
                try:
                    content_resp = await client.get(raw_url)
                    content_resp.raise_for_status()
                    return content_resp.text[:2000]
                except httpx.HTTPError:
                    continue

    return ""


def _infer_dimensions(text: str) -> list[str]:
    text_lower = text.lower()
    dims: list[str] = []

    ml_keywords = [
        "quantization",
        "model",
        "training",
        "loss",
        "bpb",
        "weight",
        "optimizer",
        "transformer",
    ]
    infra_keywords = ["gpu", "kernel", "cuda", "memory", "hardware", "benchmark"]
    math_keywords = ["entropy", "convergence", "theorem", "proof", "bound"]

    if any(kw in text_lower for kw in ml_keywords):
        dims.append("ml")
    if any(kw in text_lower for kw in infra_keywords):
        dims.append("infra")
    if any(kw in text_lower for kw in math_keywords):
        dims.append("math")

    return dims if dims else ["ml"]
