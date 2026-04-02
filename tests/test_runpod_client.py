"""Tests for compute.runpod_client — REST API client, errors, preflight."""
import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_runpod_error_carries_pod_id(self):
        from compute.runpod_client import RunPodError

        err = RunPodError("boom", pod_id="pod-123")
        assert str(err) == "boom"
        assert err.pod_id == "pod-123"

    def test_api_error_carries_status_code(self):
        from compute.runpod_client import RunPodAPIError

        err = RunPodAPIError("bad request", status_code=400, pod_id="pod-1")
        assert err.status_code == 400
        assert err.pod_id == "pod-1"

    def test_preflight_error_carries_stage(self):
        from compute.runpod_client import PreflightError

        err = PreflightError("syntax error at line 5", stage="syntax")
        assert err.stage == "syntax"
        assert err.pod_id is None

    def test_all_errors_inherit_from_runpod_error(self):
        from compute.runpod_client import (
            RunPodError,
            RunPodAPIError,
            PodReadyTimeoutError,
            SSHConnectionError,
            PreflightError,
        )

        assert issubclass(RunPodAPIError, RunPodError)
        assert issubclass(PodReadyTimeoutError, RunPodError)
        assert issubclass(SSHConnectionError, RunPodError)
        assert issubclass(PreflightError, RunPodError)


# ---------------------------------------------------------------------------
# SSH key helpers
# ---------------------------------------------------------------------------


class TestSSHKeyHelpers:
    def test_find_private_key_from_env(self, tmp_path):
        key_file = tmp_path / "my_key"
        key_file.write_text("PRIVATE KEY")
        from compute.runpod_client import _find_ssh_private_key

        with patch.dict(os.environ, {"RUNPOD_SSH_PRIVATE_KEY": str(key_file)}):
            result = _find_ssh_private_key()
        assert result == str(key_file)

    def test_find_private_key_falls_back_to_candidates(self):
        from compute.runpod_client import _find_ssh_private_key

        with patch.dict(os.environ, {"RUNPOD_SSH_PRIVATE_KEY": ""}, clear=False):
            # Should search candidates — result depends on whether keys exist on this machine
            result = _find_ssh_private_key()
        # Either a path or None, depending on the machine
        assert result is None or result.endswith(("id_ed25519", "id_rsa", "id_ecdsa"))

    def test_find_public_key_from_env(self):
        from compute.runpod_client import _find_ssh_public_key

        with patch.dict(os.environ, {"RUNPOD_SSH_PUBLIC_KEY": "ssh-ed25519 AAAA..."}):
            result = _find_ssh_public_key()
        assert result == "ssh-ed25519 AAAA..."


# ---------------------------------------------------------------------------
# REST API client
# ---------------------------------------------------------------------------


class TestRunPodClientAPI:
    """Tests for REST API methods using mocked requests."""

    def _make_client(self):
        from compute.runpod_client import RunPodClient

        return RunPodClient(api_key="test-key", template_id="tmpl-123")

    def test_create_pod_sends_correct_request(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": "pod-abc",
            "publicIp": None,
            "portMappings": None,
        }
        with patch(
            "compute.runpod_client.requests.request", return_value=mock_resp
        ) as mock_req:
            pod_id = client.create_pod()
        assert pod_id == "pod-abc"
        call_args = mock_req.call_args
        assert call_args.kwargs["method"] == "POST"
        body = call_args.kwargs["json"]
        assert body["gpuCount"] == 8
        assert "22/tcp" in body["ports"]

    def test_create_pod_raises_on_400(self):
        from compute.runpod_client import RunPodAPIError

        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        with patch(
            "compute.runpod_client.requests.request", return_value=mock_resp
        ):
            with pytest.raises(RunPodAPIError) as exc_info:
                client.create_pod()
        assert exc_info.value.status_code == 400

    def test_get_pod_returns_dict(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "pod-abc",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 17445},
        }
        with patch(
            "compute.runpod_client.requests.request", return_value=mock_resp
        ):
            pod = client.get_pod("pod-abc")
        assert pod["publicIp"] == "1.2.3.4"
        assert pod["portMappings"]["22"] == 17445

    def test_terminate_pod_sends_delete(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        with patch(
            "compute.runpod_client.requests.request", return_value=mock_resp
        ) as mock_req:
            client.terminate_pod("pod-abc")
        call_args = mock_req.call_args
        assert call_args.kwargs["method"] == "DELETE"
        assert "pod-abc" in call_args.kwargs["url"]

    def test_terminate_pod_ignores_404(self):
        from compute.runpod_client import RunPodAPIError

        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        with patch(
            "compute.runpod_client.requests.request", return_value=mock_resp
        ):
            # Should not raise
            client.terminate_pod("pod-gone")

    def test_http_retries_on_500(self):
        client = self._make_client()
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"id": "pod-abc", "desiredStatus": "RUNNING"}
        with patch(
            "compute.runpod_client.requests.request",
            side_effect=[fail_resp, fail_resp, ok_resp],
        ):
            with patch("time.sleep"):  # skip backoff delay
                pod = client.get_pod("pod-abc")
        assert pod["id"] == "pod-abc"


# ---------------------------------------------------------------------------
# Wait for ready
# ---------------------------------------------------------------------------


class TestWaitForReady:
    def _make_client(self):
        from compute.runpod_client import RunPodClient

        return RunPodClient(api_key="test-key", template_id="tmpl-123")

    def test_returns_ssh_conn_when_pod_ready(self):
        from compute.runpod_client import RunPodClient

        client = self._make_client()
        pod_response = {
            "id": "pod-abc",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 17445},
        }
        with patch.object(client, "get_pod", return_value=pod_response):
            with patch.object(RunPodClient, "_probe_ssh", return_value=True):
                ssh_conn = client.wait_for_ready("pod-abc", timeout_seconds=30)
        assert ssh_conn == "root@1.2.3.4 -p 17445"

    def test_polls_until_running(self):
        from compute.runpod_client import RunPodClient

        client = self._make_client()
        starting = {
            "id": "pod-abc",
            "desiredStatus": "CREATED",
            "publicIp": None,
            "portMappings": None,
        }
        ready = {
            "id": "pod-abc",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 17445},
        }
        with patch.object(client, "get_pod", side_effect=[starting, starting, ready]):
            with patch.object(RunPodClient, "_probe_ssh", return_value=True):
                with patch("time.sleep"):
                    ssh_conn = client.wait_for_ready(
                        "pod-abc", timeout_seconds=60
                    )
        assert ssh_conn == "root@1.2.3.4 -p 17445"

    def test_raises_timeout_when_not_ready(self):
        from compute.runpod_client import PodReadyTimeoutError

        client = self._make_client()
        starting = {
            "id": "pod-abc",
            "desiredStatus": "CREATED",
            "publicIp": None,
            "portMappings": None,
        }
        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            # First call is start, subsequent calls should exceed timeout
            return 0 if call_count <= 1 else 1000

        with patch.object(client, "get_pod", return_value=starting):
            with patch("time.time", side_effect=fake_time):
                with patch("time.sleep"):
                    with pytest.raises(PodReadyTimeoutError):
                        client.wait_for_ready("pod-abc", timeout_seconds=1)

    def test_raises_ssh_error_when_probe_fails(self):
        from compute.runpod_client import SSHConnectionError, RunPodClient

        client = self._make_client()
        ready = {
            "id": "pod-abc",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 17445},
        }
        with patch.object(client, "get_pod", return_value=ready):
            with patch.object(RunPodClient, "_probe_ssh", return_value=False):
                with patch("time.sleep"):
                    with pytest.raises(SSHConnectionError):
                        client.wait_for_ready("pod-abc", timeout_seconds=120)


# ---------------------------------------------------------------------------
# Preflight verification
# ---------------------------------------------------------------------------


class TestPreflightVerification:
    def test_valid_script_passes(self, tmp_path):
        from compute.runpod_client import verify_training_script

        script = tmp_path / "train.py"
        script.write_text("x = 1 + 1\n")
        # Should not raise
        verify_training_script(str(script))

    def test_syntax_error_raises_preflight(self, tmp_path):
        from compute.runpod_client import verify_training_script, PreflightError

        script = tmp_path / "train.py"
        script.write_text("def foo(\n")  # syntax error
        with pytest.raises(PreflightError) as exc_info:
            verify_training_script(str(script))
        assert exc_info.value.stage == "syntax"

    def test_missing_file_raises_preflight(self):
        from compute.runpod_client import verify_training_script, PreflightError

        with pytest.raises(PreflightError) as exc_info:
            verify_training_script("/nonexistent/train.py")
        assert exc_info.value.stage == "syntax"

    def test_import_error_raises_preflight(self, tmp_path):
        from compute.runpod_client import verify_training_script, PreflightError

        script = tmp_path / "train.py"
        script.write_text("import nonexistent_module_xyz_12345\n")
        with pytest.raises(PreflightError) as exc_info:
            verify_training_script(str(script), check_imports=True)
        assert exc_info.value.stage == "import"
