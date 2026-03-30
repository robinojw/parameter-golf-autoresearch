from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from research.experiments import get_current_best_bpb
from research.fetch import GradedItem, RawItem
from research.grade import _detect_harness, _run_claude, _run_opencode

VERIFIED_CACHE_PATH = Path("verified_cache.jsonl")
VERIFICATION_SCORE_THRESHOLD = 10.0
MAX_ITEMS_TO_VERIFY = 5
VERIFICATION_QUERIES_PER_ITEM = 2
SUBPROCESS_TIMEOUT_SECONDS = 180

_FULL_CONTENT_MAX_CHARS = 4000
_VERIFICATION_RESULTS_MAX_CHARS = 2000
_TECHNIQUE_NAME_MAX_CHARS = 60
_FENCE_MARKER = "```"
_NEWLINE = "\n"
_DEFAULT_SCORE = 0.0
_TIER_A_THRESHOLD = 10
_TIER_B_THRESHOLD = 7
_SEARCH_MAX_RESULTS = 3
_JSON_PREVIEW_CHARS = 200
_ENCODING = "utf-8"
_KEY_VERIFIED_SCORE = "verified_score"
_KEY_ID = "id"

VERIFICATION_PROMPT_TEMPLATE = (
    "## TASK\n"
    "You are re-evaluating a research item for the Parameter Golf challenge after deep analysis.\n"
    "You previously scored this item {original_score}/15 based on a brief abstract.\n"
    "Now you have the full content and verification evidence. Re-evaluate carefully.\n\n"
    "## CHALLENGE CONSTRAINTS (hard — violations = score 0 on size/time dimensions)\n"
    "- Artifact: train_gpt.py code bytes + zstd-compressed weights ≤ 16,000,000 bytes\n"
    "- Training: ≤ 600 seconds on 8×H100 SXM\n"
    "- No network calls or external downloads during evaluation\n"
    "- No validation data access during training\n"
    "- Metric: val_bpb on FineWeb — lower is better\n\n"
    "## CURRENT SOTA: {current_best_bpb} bpb\n\n"
    "## ORIGINAL ITEM\nTitle: {title}\nURL: {url}\n"
    "Original Score: {original_score}/15\nOriginal Score Breakdown: {score_breakdown}\n\n"
    "## FULL CONTENT\n{full_content_or_abstract}\n\n"
    "## VERIFICATION EVIDENCE\n{verification_results}\n\n"
    "## INSTRUCTIONS\n"
    "Re-evaluate this item. You now have much more context than the initial grading.\n\n"
    "Return a JSON object ONLY with these keys:\n"
    "- verified_score: float (0-15, your updated score with full context)\n"
    "- implementation_brief: string (3-5 sentences: what exactly to implement in train_gpt.py, "
    "estimated lines, specific approach, key parameters, potential pitfalls)\n"
    "- red_flags: string array (constraints violations, missing dependencies, unclear results, etc.)\n\n"
    "Be SPECIFIC in the implementation_brief — the agent will use this to write code.\n"
    "If the technique is NOT feasible under the constraints, set verified_score to 0 "
    "and explain why in red_flags.\n\n"
    "Do NOT return anything other than the JSON object. No markdown fences, no explanation."
)


@dataclass
class VerifiedItem:
    id: str
    original_score: float
    verified_score: float
    original_tier: str
    verified_tier: str
    implementation_brief: str
    verification_sources: list[str]
    full_content_available: bool
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _score_to_tier(score: float) -> str:
    if score >= _TIER_A_THRESHOLD:
        return "A"
    if score >= _TIER_B_THRESHOLD:
        return "B"
    return "C"


def _load_verified_ids() -> set[str]:
    ids: set[str] = set()
    if not VERIFIED_CACHE_PATH.exists():
        return ids
    with VERIFIED_CACHE_PATH.open("r", encoding=_ENCODING) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ids.add(json.loads(stripped)[_KEY_ID])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _append_verified(item: VerifiedItem) -> None:
    with VERIFIED_CACHE_PATH.open("a", encoding=_ENCODING) as f:
        f.write(json.dumps(asdict(item)) + _NEWLINE)
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_verified(item.id, item.verified_at)


_TITLE_STRIP_PATTERNS = [
    (r"\s*:\s*A\s+.*$", re.IGNORECASE),
    (r"\s+for\s+(Small\s+)?Language\s+Model.*$", re.IGNORECASE),
    (r"\s+in\s+Neural\s+Network.*$", re.IGNORECASE),
]


def _generate_verification_queries(title: str, abstract: str) -> list[str]:
    text = title.strip()
    for pattern, flags in _TITLE_STRIP_PATTERNS:
        text = re.sub(pattern, "", text, flags=flags)
    technique = text[:_TECHNIQUE_NAME_MAX_CHARS].strip()
    return [
        f"{technique} implementation results small language model",
        f"{technique} constraints limitations training time memory",
    ]


def _extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith(_FENCE_MARKER):
        cleaned = _NEWLINE.join(
            l for l in cleaned.splitlines() if not l.strip().startswith(_FENCE_MARKER)
        ).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # regex: outermost { } allowing one level of nested braces
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Could not extract JSON object from response: {cleaned[:_JSON_PREVIEW_CHARS]}"
    )


def _run_verification_prompt(prompt: str) -> str:
    harness = _detect_harness()
    if harness == "opencode":
        return _run_opencode(prompt)
    if harness == "claude":
        return _run_claude(prompt)
    raise RuntimeError(f"Unknown harness: {harness}")


def _find_raw_item(item_id: str, raw_items: list[RawItem]) -> RawItem | None:
    return next((item for item in raw_items if item.id == item_id), None)


async def _extract_full_content(raw: RawItem) -> tuple[str, bool]:
    from research.sources.tavily_extract import extract_url
    try:
        content = await extract_url(raw.url)
        if content:
            return content, True
    except Exception as exc:
        print(f"[verify] extract failed for {raw.url}: {exc}")
    parts = [p for p in (raw.abstract, raw.content_snippet) if p]
    return (_NEWLINE.join(parts) if parts else "(no content available)"), False


def _collect_verification_evidence(
    title: str, abstract: str
) -> tuple[list[str], list[str]]:
    from research.sources.tavily_agent import agent_search
    queries = _generate_verification_queries(title, abstract)
    results_parts: list[str] = []
    sources: list[str] = []
    for query in queries:
        try:
            result_md = agent_search(
                query, depth="basic", max_results=_SEARCH_MAX_RESULTS
            )
            results_parts.append(result_md)
            sources.extend(re.findall(r"\[.*?\]\((https?://[^\)]+)\)", result_md))
        except Exception as exc:
            print(f"[verify] search failed for query '{query}': {exc}")
            results_parts.append(f"(search failed: {exc})")
    return results_parts, sources


def _regrade_item(
    raw: RawItem, graded: GradedItem, full_content: str, evidence: str
) -> tuple[float, str]:
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(
        original_score=graded.score,
        current_best_bpb=get_current_best_bpb(),
        title=raw.title,
        url=raw.url,
        score_breakdown=json.dumps(graded.score_breakdown),
        full_content_or_abstract=full_content[:_FULL_CONTENT_MAX_CHARS],
        verification_results=evidence[:_VERIFICATION_RESULTS_MAX_CHARS],
    )
    try:
        parsed = _extract_json_object(_run_verification_prompt(prompt))
        verified_score = float(str(parsed.get(_KEY_VERIFIED_SCORE, graded.score)))
        brief = str(parsed.get("implementation_brief", ""))
        raw_flags = parsed.get("red_flags")
        red_flags: list[str] = (
            [str(f) for f in raw_flags] if isinstance(raw_flags, list) else []
        )
        if red_flags:
            brief += f" Red flags: {'; '.join(red_flags)}"
        return verified_score, brief
    except Exception as exc:
        print(f"[verify] re-grading failed for {graded.id}: {exc}")
        return (
            graded.score,
            f"Verification re-grading failed ({exc}). Original score {graded.score}/15 retained.",
        )


async def _verify_single_item(graded: GradedItem, raw: RawItem) -> VerifiedItem:
    full_content, full_content_available = await _extract_full_content(raw)
    results_parts, verification_sources = _collect_verification_evidence(
        raw.title, raw.abstract
    )
    verified_score, implementation_brief = _regrade_item(
        raw,
        graded,
        full_content,
        _NEWLINE.join(results_parts),
    )
    verified_item = VerifiedItem(
        id=graded.id,
        original_score=graded.score,
        verified_score=verified_score,
        original_tier=graded.tier,
        verified_tier=_score_to_tier(verified_score),
        implementation_brief=implementation_brief,
        verification_sources=verification_sources,
        full_content_available=full_content_available,
    )
    _append_verified(verified_item)
    return verified_item


def filter_infeasible_candidates(candidates: list[GradedItem]) -> list[GradedItem]:
    """Post-grade feasibility gate. Checks agent_summary for extractable params/bits.

    If params and bits are found and the configuration is infeasible, the item
    is dropped. If extraction fails, the item passes through (fail-open).
    Items already flagged as prefilter_rejected are dropped.
    """
    from research.extract_params import extract_params
    from compute.constraints import feasibility_report

    filtered = []
    for item in candidates:
        if "prefilter_rejected" in item.flags:
            continue

        extracted = extract_params(item.agent_summary)
        params = extracted["params"]
        bits = extracted["bits"]

        if params is not None and bits is not None:
            report = feasibility_report(params=params, bits=bits)
            if not report["feasible"]:
                print(f"[verify:gate] Dropping {item.id}: infeasible ({params:,} params at {bits}-bit)")
                continue

        filtered.append(item)

    return filtered


async def verify_top_items(
    graded_items: list[GradedItem],
    raw_items: list[RawItem],
) -> list[VerifiedItem]:
    already_verified = _load_verified_ids()
    candidates = sorted(
        [
            i
            for i in graded_items
            if i.score >= VERIFICATION_SCORE_THRESHOLD and i.id not in already_verified
        ],
        key=lambda x: x.score,
        reverse=True,
    )
    # Post-grade feasibility gate
    candidates = filter_infeasible_candidates(candidates)
    candidates = candidates[:MAX_ITEMS_TO_VERIFY]
    if not candidates:
        return []
    verified: list[VerifiedItem] = []
    for graded in candidates:
        raw = _find_raw_item(graded.id, raw_items)
        if raw is None:
            print(f"[verify] no raw item found for {graded.id}, skipping")
            continue
        try:
            verified.append(await _verify_single_item(graded, raw))
        except Exception as exc:
            print(f"[verify] failed to verify {graded.id}: {exc}")
    return verified


def get_verified_items() -> list[dict[str, object]]:
    if not VERIFIED_CACHE_PATH.exists():
        return []
    items: list[dict[str, object]] = []
    with VERIFIED_CACHE_PATH.open("r", encoding=_ENCODING) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    items.sort(
        key=lambda x: float(str(x.get(_KEY_VERIFIED_SCORE, _DEFAULT_SCORE))),
        reverse=True,
    )
    return items


async def run_verification_cycle() -> list[VerifiedItem]:
    raw_cache = Path("raw_cache.jsonl")
    graded_cache = Path("graded_cache.jsonl")
    if not raw_cache.exists() or not graded_cache.exists():
        return []

    raw_items: list[RawItem] = []
    with raw_cache.open("r", encoding=_ENCODING) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                raw_items.append(
                    RawItem(
                        id=obj["id"],
                        source=obj.get("source", ""),
                        dimension=obj.get("dimension", []),
                        title=obj.get("title", ""),
                        abstract=obj.get("abstract", ""),
                        url=obj.get("url", ""),
                        published_date=obj.get("published_date", ""),
                        content_snippet=obj.get("content_snippet", ""),
                        raw_type=obj.get("raw_type", "paper"),
                        tavily_score=float(obj.get("tavily_score", 0.0)),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    graded_items: list[GradedItem] = []
    with graded_cache.open("r", encoding=_ENCODING) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                graded_items.append(
                    GradedItem(
                        id=obj["id"],
                        score=float(obj.get("score", 0)),
                        tier=obj.get("tier", "C"),
                        score_breakdown=obj.get("score_breakdown", {}),
                        agent_summary=obj.get("agent_summary", ""),
                        flags=obj.get("flags", []),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    if not graded_items or not raw_items:
        return []

    return await verify_top_items(graded_items, raw_items)
