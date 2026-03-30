import json
from unittest.mock import patch, MagicMock
import pytest


def test_pusher_noop_when_no_url():
    """DashboardPusher is a no-op when DASHBOARD_URL is not set."""
    with patch.dict("os.environ", {}, clear=True):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()
        # Should not raise, should do nothing
        pusher.push_experiment({"id": "abc", "tier": "local"})


def test_pusher_sends_experiment():
    """DashboardPusher sends experiment_complete event."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            pusher.push_experiment({"id": "abc", "tier": "local", "val_bpb": 1.21})

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            body = json.loads(call_kwargs.kwargs.get("content", call_kwargs[1].get("content", "")))
            assert body["event"] == "experiment_complete"
            assert body["data"]["id"] == "abc"


def test_pusher_sends_heartbeat():
    """DashboardPusher sends heartbeat with agents, sota, and pipeline counts."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            pusher.push_heartbeat(
                statuses=[{"agent": "experiment", "status": "running", "last_activity": "", "restart_count": 0}],
                sota_bpb=1.05,
                pipeline_counts={"fetched": 100, "graded": 50, "verified": 10, "injected": 5},
            )

            call_kwargs = mock_post.call_args
            body = json.loads(call_kwargs.kwargs.get("content", call_kwargs[1].get("content", "")))
            assert body["event"] == "heartbeat"
            assert body["data"]["sota_bpb"] == 1.05


def test_pusher_swallows_errors():
    """DashboardPusher never raises — errors are silently swallowed."""
    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", side_effect=Exception("network down")):
            # Should not raise
            pusher.push_experiment({"id": "abc"})
