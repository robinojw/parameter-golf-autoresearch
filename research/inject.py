import json
import re
from pathlib import Path


def inject_into_program_md(
    graded_cache_path: str = "graded_cache.jsonl",
    program_md_path: str = "program.md",
    top_n: int = 12,
    max_token_budget: int = 3000,
) -> None:
    graded_items = _load_graded_sorted(graded_cache_path)
    top_items = graded_items[:top_n]

    bullets: list[str] = []
    char_budget = max_token_budget * 4
    chars_used = 0

    for item in top_items:
        flags_str = ", ".join(item.get("flags", []))
        flag_display = f"[{flags_str}] " if flags_str else ""

        bullet = (
            f"- {flag_display}**{item['title']}** — score {item['score']}/15 ({item.get('published_date', 'n/a')})\n"
            f"  {item.get('agent_summary', '')}\n"
            f"  → {item.get('url', '')}"
        )

        if chars_used + len(bullet) > char_budget:
            break

        bullets.append(bullet)
        chars_used += len(bullet)

    replacement_content = "\n".join(bullets)
    replacement = (
        f"<!-- RESEARCH_START -->\n{replacement_content}\n<!-- RESEARCH_END -->"
    )

    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- RESEARCH_START -->.*?<!-- RESEARCH_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )

    program_path.write_text(new_content)


def _load_graded_sorted(graded_cache_path: str) -> list[dict]:
    path = Path(graded_cache_path)
    if not path.exists():
        return []

    items: list[dict] = []
    raw_cache_path = Path("raw_cache.jsonl")
    raw_map: dict[str, dict] = {}

    if raw_cache_path.exists():
        with open(raw_cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    raw_map[obj["id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                graded = json.loads(line)
                raw = raw_map.get(graded.get("id", ""), {})
                graded["title"] = raw.get("title", graded.get("id", "Unknown"))
                graded["url"] = raw.get("url", "")
                graded["published_date"] = raw.get("published_date", "")
                items.append(graded)
            except (json.JSONDecodeError, KeyError):
                continue

    items.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return items
