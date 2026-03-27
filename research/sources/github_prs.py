import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from research.fetch import RawItem

WATCHED_REPOS = [
    ("openai", "parameter-golf"),
    ("KellerJordan", "modded-nanogpt"),
    ("karpathy", "autoresearch"),
]

GITHUB_API = "https://api.github.com"
COMPETITOR_SCORES_PATH = Path("competitor_scores.jsonl")
BASELINE_BPB = 1.2244

_RE_VAL_BPB_BOLD = re.compile(r"\*{0,2}val_bpb:\s*(\d+\.\d+)\*{0,2}", re.IGNORECASE)
_RE_VAL_BPB_EQ = re.compile(r"val_bpb\s*[=:]\s*(\d+\.\d+)", re.IGNORECASE)
_RE_BPB_BARE = re.compile(r"(?<![a-z_-])bpb:\s*(\d+\.\d+)", re.IGNORECASE)
_RE_BPB_SUFFIX = re.compile(r"(\d+\.\d{3,6})\s*bpb", re.IGNORECASE)

_BPB_PATTERNS = [_RE_VAL_BPB_BOLD, _RE_VAL_BPB_EQ, _RE_BPB_BARE, _RE_BPB_SUFFIX]
_TITLE_BPB_PATTERNS = [_RE_VAL_BPB_EQ, _RE_BPB_SUFFIX]


async def fetch_github_prs(since_hours: int = 48) -> list[RawItem]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    cutoff_iso = cutoff.isoformat()
    items: list[RawItem] = []
    seen_ids: set[str] = set()
    pgolf_author_map: dict[int, str] = {}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for owner, repo in WATCHED_REPOS:
            pr_items = await _fetch_prs(client, owner, repo, cutoff, seen_ids)
            items.extend(pr_items)

            is_parameter_golf = owner == "openai" and repo == "parameter-golf"
            if is_parameter_golf:
                for pr_item in pr_items:
                    record_content = await _fetch_pr_records(
                        client, owner, repo, pr_item
                    )
                    if record_content:
                        pr_item.content_snippet = record_content

                pgolf_author_map.update(
                    await _fetch_pr_authors(client, owner, repo, pr_items)
                )

                merged_items, merged_authors = await _fetch_merged_prs(
                    client, owner, repo, cutoff, seen_ids
                )
                items.extend(merged_items)
                for merged_item in merged_items:
                    record_content = await _fetch_pr_records(
                        client, owner, repo, merged_item
                    )
                    if record_content:
                        merged_item.content_snippet = record_content

                pgolf_author_map.update(merged_authors)

            commit_items = await _fetch_commits(
                client, owner, repo, cutoff_iso, seen_ids
            )
            items.extend(commit_items)

    pgolf_items = [
        item for item in items if item.id.startswith("gh:openai/parameter-golf/pr/")
    ]
    if pgolf_items:
        extract_competitor_scores(pgolf_items, pgolf_author_map)

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


async def _fetch_pr_authors(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    pr_items: list[RawItem],
) -> dict[int, str]:
    author_map: dict[int, str] = {}
    for item in pr_items:
        pr_number = _pr_number_from_id(item.id)
        if pr_number == 0:
            continue
        try:
            resp = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}",
            )
            resp.raise_for_status()
            pr_data = resp.json()
            user_data = pr_data.get("user", {}) or {}
            login = user_data.get("login", "")
            if login:
                author_map[pr_number] = login
        except (httpx.HTTPError, ValueError):
            continue
    return author_map


async def _fetch_merged_prs(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    cutoff: datetime,
    seen_ids: set[str],
) -> tuple[list[RawItem], dict[int, str]]:
    items: list[RawItem] = []
    author_map: dict[int, str] = {}
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 30,
            },
        )
        resp.raise_for_status()
        pulls = resp.json()
    except (httpx.HTTPError, ValueError):
        return items, author_map

    for pr in pulls:
        merged_at_str = pr.get("merged_at")
        if not merged_at_str:
            continue

        try:
            merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
            if merged_at < cutoff:
                continue
        except ValueError:
            continue

        pr_number = pr.get("number", 0)
        pr_id = f"gh:{owner}/{repo}/pr/{pr_number}"
        if pr_id in seen_ids:
            continue

        user_data = pr.get("user", {}) or {}
        author_login = user_data.get("login", "")
        if author_login:
            author_map[pr_number] = author_login

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

    return items, author_map


def _extract_bpb_from_content(content: str) -> float | None:
    for pattern in _BPB_PATTERNS:
        match = pattern.search(content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_bpb_from_title(title: str) -> float | None:
    for pattern in _TITLE_BPB_PATTERNS:
        match = pattern.search(title)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


_RE_TITLE_TECHNIQUE = re.compile(
    r"(?:Record|Submission|Preliminary|Non-record|Notable)[:\s]*(?:\d+\.\d+\s*(?:bpb|BPB)\s*[-—]\s*)?(.+)",
    re.IGNORECASE,
)
_TECHNIQUE_NOISE_SUFFIXES = re.compile(
    r"\s*\((?:val_bpb|3-seed|ADIITJ|std)[^)]*\)\s*$", re.IGNORECASE
)


def _extract_technique(title: str) -> str:
    pr_prefix_stripped = re.sub(
        r"^\[openai/parameter-golf\]\s*PR\s*#\d+:\s*", "", title
    )

    match = _RE_TITLE_TECHNIQUE.search(pr_prefix_stripped)
    technique = match.group(1).strip() if match else pr_prefix_stripped.strip()

    bpb_stripped = re.sub(r"\d+\.\d+\s*(?:bpb|BPB)\s*[-—]?\s*", "", technique)
    technique = _TECHNIQUE_NOISE_SUFFIXES.sub("", bpb_stripped).strip()

    technique = technique.rstrip(" -—,")
    return technique if technique else pr_prefix_stripped.strip()


def _load_existing_competitor_keys() -> set[str]:
    keys: set[str] = set()
    if not COMPETITOR_SCORES_PATH.exists():
        return keys
    with open(COMPETITOR_SCORES_PATH, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pr_num = obj.get("pr_number", 0)
                technique = obj.get("technique", "")
                keys.add(f"{pr_num}:{technique}")
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _pr_number_from_id(item_id: str) -> int:
    try:
        return int(item_id.split("/pr/")[-1])
    except (ValueError, IndexError):
        return 0


def extract_competitor_scores(
    pr_items: list[RawItem], author_map: dict[int, str]
) -> None:
    existing_keys = _load_existing_competitor_keys()
    new_records: list[dict[str, object]] = []

    for item in pr_items:
        pr_number = _pr_number_from_id(item.id)
        if pr_number == 0:
            continue

        val_bpb = (
            _extract_bpb_from_content(item.content_snippet)
            if item.content_snippet
            else None
        )
        if val_bpb is None:
            val_bpb = _extract_bpb_from_title(item.title)
        if val_bpb is None:
            continue

        technique = _extract_technique(item.title)
        dedup_key = f"{pr_number}:{technique}"
        if dedup_key in existing_keys:
            continue

        author = author_map.get(pr_number, "")
        url = item.url
        delta_from_baseline = round(val_bpb - BASELINE_BPB, 6)

        record = {
            "pr_number": pr_number,
            "author": author,
            "title": item.title,
            "val_bpb": val_bpb,
            "technique": technique,
            "delta_from_baseline": delta_from_baseline,
            "url": url,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }

        new_records.append(record)
        existing_keys.add(dedup_key)

    if not new_records:
        return

    with open(COMPETITOR_SCORES_PATH, "a") as fh:
        for record in new_records:
            fh.write(json.dumps(record) + "\n")
