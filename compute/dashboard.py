"""Fire-and-forget push client for the Parameter Golf dashboard."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

_TIMEOUT_SECONDS = 5


class DashboardPusher:
    """Pushes events to the CF Workers dashboard.

    If DASHBOARD_URL is not set, all methods are silent no-ops.
    All methods swallow exceptions — the dashboard being down never blocks agents.
    """

    def __init__(self) -> None:
        url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
        # Ensure the URL includes the /pgolf base path
        if url and not url.endswith("/pgolf"):
            url = url + "/pgolf"
        self._url = url
        self._token = os.environ.get("DASHBOARD_TOKEN", "")

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def _post(self, event: str, data: Any) -> None:
        if not self.enabled:
            return
        try:
            httpx.post(
                f"{self._url}/api/ingest",
                content=json.dumps({"event": event, "data": data}),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=_TIMEOUT_SECONDS,
            )
        except Exception:
            pass

    def push_experiment(self, row: dict) -> None:
        self._post("experiment_complete", row)

    def push_research(self, items: list[dict]) -> None:
        self._post("research_graded", items)

    def push_verified(self, item_id: str, verified_at: str) -> None:
        self._post("research_verified", {"id": item_id, "verified_at": verified_at})

    def push_budget(self, snapshot: dict, run: dict | None = None) -> None:
        self._post("budget_update", {"snapshot": snapshot, "run": run})

    def push_doc(self, key: str, content: str) -> None:
        self._post("doc_update", {"key": key, "content": content})

    def push_heartbeat(
        self,
        statuses: list[dict],
        sota_bpb: float,
        pipeline_counts: dict,
    ) -> None:
        self._post(
            "heartbeat",
            {
                "agents": statuses,
                "sota_bpb": sota_bpb,
                "pipeline_counts": pipeline_counts,
            },
        )
