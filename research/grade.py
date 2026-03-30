import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from research.experiments import (
    get_competitor_scores,
    get_current_best_bpb,
    get_failed_experiments,
    get_proven_techniques,
)
from research.fetch import GradedItem, RawItem

GRADED_CACHE_PATH = Path("graded_cache.jsonl")
BATCH_SIZE = 10
SUBPROCESS_TIMEOUT_SECONDS = 120
HARNESS_OPENCODE = "opencode"
HARNESS_CLAUDE = "claude"
HARNESS_AUTO = "auto"
_KEY_ID = "id"
_KEY_TEXT = "text"
_KEY_JSON = "json"
_FENCE_MARKER = "```"
_PROMPT_SEPARATOR = "\n\n---\n\nItems to grade:\n"
_PAYLOAD_INDENT = 2
_ERROR_SCORE = 0.0
_NEWLINE = "\n"

GRADING_PROMPT_TEMPLATE = """You are a research grading assistant for a competitive ML challenge.

## CHALLENGE CONSTRAINTS (hard — violations = score 0 on size/time dimensions)
- Artifact: train_gpt.py code bytes + zstd-compressed weights ≤ 16,000,000 bytes
- Training: ≤ 600 seconds on 8×H100 SXM
- No network calls or external downloads during evaluation
- No validation data access during training
- Metric: val_bpb on FineWeb — lower is better

## CURRENT LEADERBOARD SOTA: {current_best_bpb} bpb

## PROVEN TECHNIQUES (already on leaderboard — penalize re-implementing):
{proven_techniques}
{failed_experiments_section}{competitor_scores_section}
## SCORING RUBRIC (total /{max_score}):
1. bpb_impact (0-3): evidence this reduces validation loss / bits per byte
2. size_compatibility (0-3): fits or reduces the 16MB artifact constraint
3. time_compatibility (0-2): does not exceed 10-minute training budget
4. implementability (0-4): implementable in train_gpt.py in <100 lines, no new deps
5. novelty (0-3): not already on leaderboard or in competitor submissions; opens new search direction
{competitor_validated_dimension}
Return a JSON array only — one object per item with these keys:
id, score, score_breakdown, agent_summary (2-3 sentences), flags (string array)

Do NOT return anything other than the JSON array. No markdown fences, no explanation."""

_MAX_FAILED_IN_PROMPT = 10
_MAX_COMPETITORS_IN_PROMPT = 10
_SCORE_BASE = 15
_SCORE_WITH_COMPETITORS = 17
_TIER_A_WITH_COMPETITORS = 12
_TIER_B_WITH_COMPETITORS = 9


def _build_grading_prompt() -> str:
    """Build the grading prompt dynamically from experiment history."""
    current_best = get_current_best_bpb()
    proven = get_proven_techniques()
    failed = get_failed_experiments()
    competitors = get_competitor_scores()

    proven_text = ", ".join(proven)

    if failed:
        failed_lines = []
        for exp in failed[-_MAX_FAILED_IN_PROMPT:]:
            failed_lines.append(
                f"- {exp['description']} (val_bpb={exp['val_bpb']}, tier={exp['tier']})"
            )
        failed_section = (
            "\n## FAILED EXPERIMENTS (penalize re-attempts of these approaches):\n"
            + _NEWLINE.join(failed_lines)
            + "\n"
        )
    else:
        failed_section = ""

    has_competitors = bool(competitors)
    if has_competitors:
        comp_lines = []
        for comp in competitors[:_MAX_COMPETITORS_IN_PROMPT]:
            comp_lines.append(
                f"- PR #{comp['pr_number']} by @{comp['author']}: "
                f"{comp['technique']} → {comp['val_bpb']} bpb "
                f"(Δ{comp['delta_from_baseline']:+.4f})"
            )
        competitor_section = (
            "\n## COMPETITOR SCORES (what others achieved and with what techniques):\n"
            + _NEWLINE.join(comp_lines)
            + "\n"
        )
    else:
        competitor_section = ""

    max_score = _SCORE_WITH_COMPETITORS if has_competitors else _SCORE_BASE
    competitor_dim = (
        "6. competitor_validated (0-2): technique validated by competitors with measurable bpb improvement\n"
        if has_competitors
        else ""
    )

    return GRADING_PROMPT_TEMPLATE.format(
        current_best_bpb=current_best,
        proven_techniques=proven_text,
        failed_experiments_section=failed_section,
        competitor_scores_section=competitor_section,
        max_score=max_score,
        competitor_validated_dimension=competitor_dim,
    )


ABSTRACT_TRUNCATE = 800
SNIPPET_TRUNCATE = 300
TIER_A_THRESHOLD = 10
TIER_B_THRESHOLD = 7


def _detect_harness() -> str:
    configured = os.environ.get("GRADING_HARNESS", HARNESS_AUTO).lower()
    if configured != HARNESS_AUTO:
        return configured

    if shutil.which(HARNESS_OPENCODE):
        return HARNESS_OPENCODE
    if shutil.which(HARNESS_CLAUDE):
        return HARNESS_CLAUDE

    raise RuntimeError(
        "No coding agent found. Install opencode or claude code, "
        "or set GRADING_HARNESS=opencode|claude in .env"
    )


def _run_opencode(prompt: str) -> str:
    result = subprocess.run(
        [HARNESS_OPENCODE, "run", prompt, "--format", _KEY_JSON],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"opencode failed (exit {result.returncode}): {result.stderr}"
        )

    text_parts: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if event.get("type") != _KEY_TEXT:
            continue
        part_text = event.get("part", {}).get(_KEY_TEXT, "")
        if part_text:
            text_parts.append(part_text)

    if not text_parts:
        raise RuntimeError("opencode returned no text events")
    return "".join(text_parts)


def _run_claude(prompt: str) -> str:
    result = subprocess.run(
        [HARNESS_CLAUDE, "-p", prompt, "--output-format", _KEY_JSON, "--bare"],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude failed (exit {result.returncode}): {result.stderr}")

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"claude returned invalid JSON: {exc}") from exc

    text = output.get("result", "")
    if not text:
        raise RuntimeError("claude returned empty result")
    return text


def _run_grading_prompt(items_json: str) -> str:
    harness = _detect_harness()
    grading_prompt = _build_grading_prompt()
    full_prompt = f"{grading_prompt}\n\n---\n\nItems to grade:\n{items_json}"

    if harness == HARNESS_OPENCODE:
        return _run_opencode(full_prompt)
    if harness == HARNESS_CLAUDE:
        return _run_claude(full_prompt)
    raise RuntimeError(f"Unknown harness: {harness}")


def _extract_json_array(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith(_FENCE_MARKER):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith(_FENCE_MARKER)]
        cleaned = _NEWLINE.join(lines).strip()

    parsed = json.loads(cleaned)

    if isinstance(parsed, dict):
        parsed = parsed.get("items", parsed.get("results", []))
    if not isinstance(parsed, list):
        parsed = [parsed]
    return parsed


def _build_batch_payload(batch: list[RawItem]) -> str:
    payload = []
    for item in batch:
        payload.append(
            {
                _KEY_ID: item.id,
                "source": item.source,
                "title": item.title,
                "abstract": item.abstract[:ABSTRACT_TRUNCATE],
                "url": item.url,
                "raw_type": item.raw_type,
                "content_snippet": item.content_snippet[:SNIPPET_TRUNCATE],
            }
        )
    return json.dumps(payload, indent=_PAYLOAD_INDENT)


def _score_to_tier(score: float, has_competitors: bool = False) -> str:
    if has_competitors:
        if score >= _TIER_A_WITH_COMPETITORS:
            return "A"
        if score >= _TIER_B_WITH_COMPETITORS:
            return "B"
        return "C"
    if score >= TIER_A_THRESHOLD:
        return "A"
    if score >= TIER_B_THRESHOLD:
        return "B"
    return "C"


def _make_error_item(item_id: str, reason: str) -> GradedItem:
    return GradedItem(
        id=item_id,
        score=_ERROR_SCORE,
        tier="C",
        score_breakdown={},
        agent_summary=reason,
        flags=["grading_error"],
        grade_error=True,
    )


def prefilter_infeasible(items: list[RawItem]) -> dict:
    """Pre-filter items that are mathematically infeasible.

    Attempts to extract params and bits from title + abstract.
    If both are found and feasibility_report says infeasible, reject.
    If extraction fails, pass through (fail-open).

    Returns:
        {"passed": [...], "rejected": [...]}
    """
    from research.extract_params import extract_params
    from compute.constraints import feasibility_report

    passed = []
    rejected = []

    for item in items:
        text = f"{item.title} {item.abstract}"
        extracted = extract_params(text)
        params = extracted["params"]
        bits = extracted["bits"]

        if params is not None and bits is not None:
            report = feasibility_report(params=params, bits=bits)
            if not report["feasible"]:
                rejected.append(item)
                continue

        passed.append(item)

    if rejected:
        print(f"[grade:prefilter] Rejected {len(rejected)} infeasible items")

    return {"passed": passed, "rejected": rejected}


def grade_items(items: list[RawItem]) -> list[GradedItem]:
    already_graded = _load_graded_ids()
    to_grade = [item for item in items if item.id not in already_graded]

    if not to_grade:
        return []

    # Pre-filter: reject mathematically infeasible items before LLM grading
    filter_result = prefilter_infeasible(to_grade)
    to_grade = filter_result["passed"]
    rejected_graded = [
        GradedItem(
            id=item.id, score=0, tier="C",
            score_breakdown={}, agent_summary="Auto-rejected: infeasible constraints",
            flags=["prefilter_rejected"],
        )
        for item in filter_result["rejected"]
    ]

    if not to_grade:
        if rejected_graded:
            _append_graded(rejected_graded)
        return rejected_graded

    has_competitors = bool(get_competitor_scores())
    graded: list[GradedItem] = []

    for batch_start in range(0, len(to_grade), BATCH_SIZE):
        batch = to_grade[batch_start : batch_start + BATCH_SIZE]
        batch_json = _build_batch_payload(batch)

        try:
            response_text = _run_grading_prompt(batch_json)
            parsed = _extract_json_array(response_text)
            response_map = {
                entry[_KEY_ID]: entry
                for entry in parsed
                if isinstance(entry, dict) and _KEY_ID in entry
            }

            for item in batch:
                graded_result = response_map.get(item.id)
                if graded_result:
                    score = float(graded_result.get("score", 0))
                    graded.append(
                        GradedItem(
                            id=item.id,
                            score=score,
                            tier=_score_to_tier(score, has_competitors),
                            score_breakdown=graded_result.get("score_breakdown", {}),
                            agent_summary=graded_result.get("agent_summary", ""),
                            flags=graded_result.get("flags", []),
                        )
                    )
                else:
                    graded.append(
                        _make_error_item(
                            item.id, "Grading response missing for this item."
                        )
                    )

        except Exception as exc:
            print(f"[grade] batch failed: {exc}")
            for item in batch:
                graded.append(_make_error_item(item.id, f"Grading failed: {exc}"))

    graded.extend(rejected_graded)
    _append_graded(graded)
    return graded


def _load_graded_ids() -> set[str]:
    ids: set[str] = set()
    if not GRADED_CACHE_PATH.exists():
        return ids
    with open(GRADED_CACHE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj[_KEY_ID])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _append_graded(items: list[GradedItem]) -> None:
    if not items:
        return
    with open(GRADED_CACHE_PATH, "a") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + _NEWLINE)
