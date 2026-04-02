"""RunPod REST API client with direct SSH and preflight verification.

Pod lifecycle via https://rest.runpod.io/v1/pods.
SSH via direct public IP (not the RunPod proxy).
"""

from __future__ import annotations

import ast
import atexit
import os
import pathlib
import signal
import subprocess
import textwrap
import time

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://rest.runpod.io/v1"
_DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
_DEFAULT_GPU_COUNT = 8
_DEFAULT_VOLUME_GB = 50
_DEFAULT_TIMEOUT_SECONDS = 900
_POLL_INTERVAL = 5
_SSH_PROBE_TIMEOUT_SECONDS = 30
_SSH_CONNECT_TIMEOUT_SECONDS = 20
_SSH_PROBE_RETRIES = 5
_SSH_PROBE_RETRY_DELAY = 10
_HTTP_TIMEOUT = 30
_HTTP_RETRIES = 3
_HTTP_BACKOFF_BASE = 1  # seconds, doubles each retry
_PREFLIGHT_TIMEOUT_SECONDS = 30

# HTTP-based results polling (git-clone workflow)
_RESULTS_HTTP_PORT = 18080
_RESULTS_POLL_INTERVAL = 30
_RESULTS_POLL_TIMEOUT = 2700  # 45 min max (training is 600s + overhead)
_GIT_REPO = "robinojw/parameter-golf-autoresearch"

_SSH_BIN = "ssh"
_PORT_FLAG = "-p"
_OPT_FLAG = "-o"

_SSH_KEY_CANDIDATES = [
    "~/.ssh/id_ed25519",
    "~/.ssh/id_rsa",
    "~/.ssh/id_ecdsa",
]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunPodError(Exception):
    """Base for all RunPod errors. Carries pod_id for cleanup."""

    def __init__(self, message: str, pod_id: str | None = None):
        super().__init__(message)
        self.pod_id = pod_id


class RunPodAPIError(RunPodError):
    """REST API returned a non-2xx status."""

    def __init__(self, message: str, status_code: int, pod_id: str | None = None):
        super().__init__(message, pod_id)
        self.status_code = status_code


class PodReadyTimeoutError(RunPodError):
    """Pod did not reach a ready state within the timeout."""


class SSHConnectionError(RunPodError):
    """SSH probe or command failed after retries."""


class PreflightError(RunPodError):
    """Pre-promotion validation of training script failed."""

    def __init__(self, message: str, stage: str):
        super().__init__(message, pod_id=None)
        self.stage = stage  # "syntax", "import", or "model_init"


# ---------------------------------------------------------------------------
# SSH key helpers
# ---------------------------------------------------------------------------


def _find_ssh_private_key() -> str | None:
    """Find the SSH private key file matching the public key used for RunPod."""
    explicit_key = os.environ.get("RUNPOD_SSH_PRIVATE_KEY", "")
    if explicit_key and pathlib.Path(explicit_key).expanduser().exists():
        return str(pathlib.Path(explicit_key).expanduser())
    for key_file in _SSH_KEY_CANDIDATES:
        key_path = pathlib.Path(key_file).expanduser()
        if key_path.exists():
            return str(key_path)
    return None


def _find_ssh_public_key() -> str:
    """Find the SSH public key to inject into RunPod pods."""
    pub_key = os.environ.get("RUNPOD_SSH_PUBLIC_KEY", "")
    if pub_key:
        return pub_key
    for key_file in _SSH_KEY_CANDIDATES:
        pub_path = pathlib.Path(f"{key_file}.pub").expanduser()
        if pub_path.exists():
            return pub_path.read_text().strip()
    return ""


# ---------------------------------------------------------------------------
# REST API helpers
# ---------------------------------------------------------------------------


def _api_request(
    method: str,
    path: str,
    api_key: str,
    json: dict | None = None,
    pod_id: str | None = None,
) -> dict:
    """Make an authenticated request to the RunPod REST API with retry on 5xx."""
    url = f"{_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_exc: Exception | None = None

    for attempt in range(_HTTP_RETRIES):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _HTTP_RETRIES - 1:
                time.sleep(_HTTP_BACKOFF_BASE * (2**attempt))
                continue
            raise RunPodAPIError(
                f"Request failed after {_HTTP_RETRIES} attempts: {exc}",
                status_code=0,
                pod_id=pod_id,
            ) from exc

        if resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return {}

        # 4xx: client error, don't retry
        if 400 <= resp.status_code < 500:
            raise RunPodAPIError(
                f"{method} {path} returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                pod_id=pod_id,
            )

        # 5xx: server error, retry with backoff
        last_exc = RunPodAPIError(
            f"{method} {path} returned {resp.status_code}: {resp.text[:200]}",
            status_code=resp.status_code,
            pod_id=pod_id,
        )
        if attempt < _HTTP_RETRIES - 1:
            delay = _HTTP_BACKOFF_BASE * (2**attempt)
            print(f"  API {resp.status_code}, retrying in {delay}s...")
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RunPodClient:
    def __init__(self, api_key: str, template_id: str) -> None:
        self.api_key = api_key
        self.template_id = template_id
        self._active_pods: set[str] = set()
        atexit.register(self._cleanup_all)
        signal.signal(signal.SIGTERM, lambda s, f: self._cleanup_all())

    def create_pod(
        self,
        gpu_count: int = _DEFAULT_GPU_COUNT,
        gpu_type: str = _DEFAULT_GPU_TYPE,
    ) -> str:
        """Create a pod and return its ID."""
        pub_key = _find_ssh_public_key()
        env = {}
        if pub_key:
            env["PUBLIC_KEY"] = pub_key

        body = {
            "name": f"pgolf-{int(time.time())}",
            "imageName": "runpod/parameter-golf:latest",
            "gpuTypeIds": [gpu_type],
            "gpuCount": gpu_count,
            "templateId": self.template_id,
            "volumeInGb": _DEFAULT_VOLUME_GB,
            "ports": ["22/tcp"],
            "env": env,
        }

        result = _api_request("POST", "/pods", self.api_key, json=body)
        pod_id = result["id"]
        self._active_pods.add(pod_id)
        print(f"Launched pod {pod_id} ({gpu_count}x {gpu_type})")
        return pod_id

    def get_pod(self, pod_id: str) -> dict:
        """Get current pod status."""
        return _api_request("GET", f"/pods/{pod_id}", self.api_key, pod_id=pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod. Idempotent — ignores 404."""
        try:
            _api_request("DELETE", f"/pods/{pod_id}", self.api_key, pod_id=pod_id)
        except RunPodAPIError as exc:
            if exc.status_code != 404:
                raise
        self._active_pods.discard(pod_id)
        print(f"Terminated pod {pod_id}")

    def wait_for_ready(
        self, pod_id: str, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    ) -> str:
        """Poll until pod has a public IP with SSH, then probe. Returns ssh_conn string."""
        start = time.time()
        ssh_conn: str | None = None

        # Phase 1: poll API until pod is RUNNING with public IP + port mapping
        while time.time() - start < timeout_seconds:
            pod = self.get_pod(pod_id)
            status = pod.get("desiredStatus", "")
            public_ip = pod.get("publicIp") or ""
            port_mappings = pod.get("portMappings") or {}
            ssh_port = port_mappings.get("22") or port_mappings.get(22)

            if status == "RUNNING" and public_ip and ssh_port:
                ssh_conn = f"root@{public_ip} {_PORT_FLAG} {ssh_port}"
                break

            print(f"Pod {pod_id} status={status} ip={public_ip or 'pending'}...")
            time.sleep(_POLL_INTERVAL)
        else:
            raise PodReadyTimeoutError(
                f"Pod {pod_id} not ready after {timeout_seconds}s",
                pod_id=pod_id,
            )

        # Phase 2: SSH probe with retries
        print(f"Pod {pod_id} API ready: {ssh_conn}. Probing SSH...")
        for attempt in range(1, _SSH_PROBE_RETRIES + 1):
            if self._probe_ssh(ssh_conn):
                print(f"Pod {pod_id} SSH ready: {ssh_conn}")
                return ssh_conn
            if attempt < _SSH_PROBE_RETRIES:
                print(
                    f"  SSH probe {attempt}/{_SSH_PROBE_RETRIES} failed, "
                    f"retrying in {_SSH_PROBE_RETRY_DELAY}s..."
                )
                time.sleep(_SSH_PROBE_RETRY_DELAY)

        raise SSHConnectionError(
            f"SSH probe failed after {_SSH_PROBE_RETRIES} attempts for {ssh_conn}",
            pod_id=pod_id,
        )

    @staticmethod
    def _probe_ssh(ssh_conn: str) -> bool:
        """Test SSH connectivity. Returns True if reachable."""
        parts = ssh_conn.split()
        user_host = parts[0]
        port = parts[parts.index(_PORT_FLAG) + 1]
        cmd = [
            _SSH_BIN,
            _PORT_FLAG,
            port,
            _OPT_FLAG,
            "StrictHostKeyChecking=no",
            _OPT_FLAG,
            f"ConnectTimeout={_SSH_CONNECT_TIMEOUT_SECONDS}",
            _OPT_FLAG,
            "BatchMode=yes",
        ]
        identity_file = _find_ssh_private_key()
        if identity_file:
            cmd.extend(["-i", identity_file])
        cmd.extend([user_host, "echo ok"])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SSH_PROBE_TIMEOUT_SECONDS,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _build_training_script(self, run_id: str) -> str:
        """Build a self-contained bash training script for baked-in pod execution."""
        return (
            "#!/bin/bash\n"
            "set -e\n"
            'echo "PGOLF_START run_id=$RUN_ID"\n'
            "\n"
            "python3 -m pip install -q zstandard --break-system-packages 2>&1 | grep -v 'already satisfied' || true\n"
            "\n"
            "echo 'Installing Flash Attention 3 (Hopper kernels)...'\n"
            "pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper --no-build-isolation --break-system-packages 2>&1 | tail -3 || true\n"
            "python3 -c 'from flash_attn_interface import flash_attn_func; print(\"FA3 OK\")' 2>&1 || echo 'FA3 install failed, falling back to FA2'\n"
            "\n"
            'git clone --depth 1 --branch "$GIT_BRANCH" '
            '"https://${GITHUB_TOKEN}@github.com/' + _GIT_REPO + '.git" '
            "/workspace/repo\n"
            "cp /workspace/repo/train_gpt.py /workspace/train_gpt.py\n"
            "cp -r /workspace/repo/data /workspace/data 2>/dev/null || true\n"
            "\n"
            "_dp=\n"
            "for _try in \\\n"
            "    /data/datasets/fineweb10B_sp1024 \\\n"
            "    /workspace/data/datasets/fineweb10B_sp1024 \\\n"
            "    /workspace/datasets/fineweb10B_sp1024 \\\n"
            "    /opt/datasets/fineweb10B_sp1024 \\\n"
            "    /workspace/repo/data/datasets/fineweb10B_sp1024; do\n"
            '    if ls "$_try"/fineweb_train_000001.bin 2>/dev/null; then _dp=$_try; break; fi\n'
            "done\n"
            "\n"
            'if [ -z "$_dp" ]; then\n'
            '    echo "Full dataset not pre-installed, downloading 80 shards from HuggingFace..."\n'
            "    cd /workspace/repo/data && python3 cached_challenge_fineweb.py --train-shards 80 || true\n"
            "    cd /workspace\n"
            "    _dp=/workspace/repo/data/datasets/fineweb10B_sp1024\n"
            "fi\n"
            "\n"
            "export DATA_PATH=$_dp\n"
            '_nshards=$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)\n'
            'echo "DATA_PATH=$DATA_PATH (train shards: $_nshards)"\n'
            'if [ "$_nshards" -ge 4 ]; then export LOADER_MODE=coprime; '
            "else export LOADER_MODE=sequential; fi\n"
            'echo "LOADER_MODE=$LOADER_MODE"\n'
            "\n"
            'echo "PGOLF_ENV_CONFIG_START"\n'
            'echo "NPROC=$NPROC"\n'
            'echo "DATA_PATH=$DATA_PATH"\n'
            'echo "LOADER_MODE=$LOADER_MODE"\n'
            'echo "GIT_BRANCH=$GIT_BRANCH"\n'
            'echo "RUN_ID=$RUN_ID"\n'
            'python3 -c "from flash_attn_interface import flash_attn_func; print(\'FA3 OK\')" 2>/dev/null || echo "FA3 NOT_AVAILABLE"\n'
            "python3 -c \"import brotli; print('compressor:brotli')\" 2>/dev/null || "
            "python3 -c \"import zstandard; print('compressor:zstd')\" 2>/dev/null || "
            'echo "compressor:zlib"\n'
            'echo "PGOLF_ENV_CONFIG_END"\n'
            "\n"
            "cd /workspace\n"
            "set +e\n"
            "RUN_ID=$RUN_ID torchrun --standalone --nproc_per_node=${NPROC:-8} train_gpt.py 2>&1 | tee /workspace/run.log\n"
            "EXIT_CODE=${PIPESTATUS[0]}\n"
            "set -e\n"
            'echo "PGOLF_TRAINING_EXIT_CODE=$EXIT_CODE"\n'
            "\n"
            "mkdir -p /workspace/results\n"
            "cp /workspace/run.log /workspace/results/ 2>/dev/null || true\n"
            "cp /workspace/final_model.int6.zst /workspace/results/model.zst 2>/dev/null || true\n"
            "cp /workspace/final_model.pt /workspace/results/model.bin 2>/dev/null || true\n"
            'echo "{\\"exit_code\\": $EXIT_CODE, \\"completed\\": true}" > /workspace/results/results.json\n'
            "\n"
            "cd /workspace/results\n"
            'echo "PGOLF_HTTP_READY port=' + str(_RESULTS_HTTP_PORT) + '"\n'
            "python3 -m http.server " + str(_RESULTS_HTTP_PORT) + " &\n"
            "\n"
            "sleep 600\n"
        )

    def create_training_pod(
        self,
        run_id: str,
        git_branch: str,
        env_vars: dict[str, str] | None = None,
        gpu_count: int = _DEFAULT_GPU_COUNT,
        gpu_type: str = _DEFAULT_GPU_TYPE,
    ) -> str:
        """Create a pod that runs training autonomously via git-clone (no SSH needed)."""
        pub_key = _find_ssh_public_key()
        startup_script = self._build_training_script(run_id)

        env: dict[str, str] = {}
        if pub_key:
            env["PUBLIC_KEY"] = pub_key

        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            env["GITHUB_TOKEN"] = github_token
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        env["GIT_BRANCH"] = git_branch
        env["RUN_ID"] = run_id
        env["NPROC"] = str(gpu_count)

        if env_vars:
            env.update(env_vars)

        body = {
            "name": f"pgolf-{run_id}",
            "imageName": "runpod/parameter-golf:latest",
            "gpuTypeIds": [gpu_type],
            "gpuCount": gpu_count,
            "templateId": self.template_id,
            "volumeInGb": _DEFAULT_VOLUME_GB,
            "ports": ["22/tcp", f"{_RESULTS_HTTP_PORT}/http"],
            "env": env,
            "dockerEntrypoint": ["/bin/bash", "-c"],
            "dockerStartCmd": [startup_script],
        }

        result = _api_request("POST", "/pods", self.api_key, json=body)
        pod_id = result["id"]
        self._active_pods.add(pod_id)
        print(
            f"Launched training pod {pod_id} ({gpu_count}x {gpu_type}, branch={git_branch})"
        )
        return pod_id

    def _results_url(self, pod_id: str, filename: str = "") -> str:
        base = f"https://{pod_id}-{_RESULTS_HTTP_PORT}.proxy.runpod.net"
        if filename:
            return f"{base}/{filename}"
        return base

    def wait_for_results(
        self,
        pod_id: str,
        timeout_seconds: int = _RESULTS_POLL_TIMEOUT,
    ) -> dict:
        """Poll the HTTP results endpoint until training completes. Returns results dict."""
        url = self._results_url(pod_id, "results.json")
        start = time.time()
        last_status = ""

        while time.time() - start < timeout_seconds:
            try:
                resp = requests.get(url, timeout=_HTTP_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("completed"):
                        print(f"Training completed: exit_code={data.get('exit_code')}")
                        return data
                new_status = f"HTTP {resp.status_code}"
            except requests.ConnectionError:
                new_status = "connection_error"
            except requests.Timeout:
                new_status = "timeout"
            except requests.RequestException as exc:
                new_status = f"error: {exc}"

            if new_status != last_status:
                elapsed = int(time.time() - start)
                print(f"  Polling {pod_id} results: {new_status} ({elapsed}s elapsed)")
                last_status = new_status

            time.sleep(_RESULTS_POLL_INTERVAL)

        raise PodReadyTimeoutError(
            f"Training results not available after {timeout_seconds}s",
            pod_id=pod_id,
        )

    def download_result_file(
        self,
        pod_id: str,
        remote_filename: str,
        local_path: str,
    ) -> bool:
        """Download a file from the pod's HTTP results server. Returns True if successful."""
        url = self._results_url(pod_id, remote_filename)
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 404:
                print(f"  Result file not found: {remote_filename}")
                return False
            resp.raise_for_status()
            pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            size_mb = pathlib.Path(local_path).stat().st_size / (1024 * 1024)
            print(f"  Downloaded {remote_filename} ({size_mb:.1f} MB)")
            return True
        except requests.RequestException as exc:
            print(f"  Failed to download {remote_filename}: {exc}")
            return False

    def _cleanup_all(self) -> None:
        """Terminate all running pods — both tracked in-memory and orphaned from killed processes."""
        terminated = 0
        try:
            resp = requests.get(
                f"{_BASE_URL}/pods",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            for pod in resp.json():
                if pod.get("desiredStatus") == "RUNNING":
                    pod_id = pod["id"]
                    pod_name = pod.get("name", "")
                    if pod_name.startswith("pgolf-"):
                        try:
                            _api_request("DELETE", f"/pods/{pod_id}", self.api_key)
                            print(f"  Terminated orphaned pod {pod_id} ({pod_name})")
                            terminated += 1
                        except Exception as exc:
                            print(f"  Failed to terminate {pod_id}: {exc}")
        except Exception as exc:
            print(f"  Failed to list pods for cleanup: {exc}")

        for pod_id in list(self._active_pods):
            try:
                _api_request("DELETE", f"/pods/{pod_id}", self.api_key)
                print(f"  Terminated tracked pod {pod_id}")
                terminated += 1
            except Exception:
                pass
        self._active_pods.clear()

        if terminated:
            print(f"Cleaned up {terminated} pod(s)")


# ---------------------------------------------------------------------------
# Preflight verification
# ---------------------------------------------------------------------------


def verify_training_script(
    script_path: str,
    python_bin: str | None = None,
    check_imports: bool = False,
    check_model: bool = False,
) -> None:
    """Verify training script before launching a pod.

    Stages run in order; each raises PreflightError on failure:
      1. syntax  — ast.parse the source file
      2. import  — import the module in a subprocess (if check_imports=True)
      3. model_init — instantiate Hyperparameters + GPT on CPU (if check_model=True)
    """
    # Stage 1: Syntax
    try:
        source = pathlib.Path(script_path).read_text(encoding="utf-8")
        ast.parse(source, filename=script_path)
    except FileNotFoundError:
        raise PreflightError(f"Script not found: {script_path}", stage="syntax")
    except SyntaxError as exc:
        raise PreflightError(
            f"Syntax error in {script_path}: {exc}", stage="syntax"
        ) from exc

    if not check_imports and not check_model:
        return

    # Resolve python binary
    if python_bin is None:
        venv = pathlib.Path(__file__).parent.parent / ".venv" / "bin" / "python"
        python_bin = str(venv) if venv.exists() else "python3"

    # Stage 2: Import check
    if check_imports or check_model:
        import_code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('_check', {script_path!r}); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod); "
            "print('import_ok')"
        )
        try:
            result = subprocess.run(
                [python_bin, "-c", import_code],
                capture_output=True,
                text=True,
                timeout=_PREFLIGHT_TIMEOUT_SECONDS,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[-500:]
                raise PreflightError(
                    f"Import check failed for {script_path}:\n{stderr}",
                    stage="import",
                )
        except subprocess.TimeoutExpired:
            raise PreflightError(
                f"Import check timed out after {_PREFLIGHT_TIMEOUT_SECONDS}s",
                stage="import",
            )

    # Stage 3: Model init
    if check_model:
        script_dir = str(pathlib.Path(script_path).parent)
        model_code = textwrap.dedent(f"""\
            import sys, os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            sys.path.insert(0, {script_dir!r})
            from train_gpt import Hyperparameters, GPT
            args = Hyperparameters()
            model = GPT(
                vocab_size=args.vocab_size, num_layers=args.num_layers,
                model_dim=args.model_dim, num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads, mlp_mult=int(args.mlp_mult),
                tie_embeddings=args.tie_embeddings,
                tied_embed_init_std=args.tied_embed_init_std,
                logit_softcap=args.logit_softcap, rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init,
            )
            n = sum(p.numel() for p in model.parameters())
            print(f"model_ok params={{n}}")
        """)
        try:
            result = subprocess.run(
                [python_bin, "-c", model_code],
                capture_output=True,
                text=True,
                timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[-500:]
                raise PreflightError(
                    f"Model init failed:\n{stderr}",
                    stage="model_init",
                )
        except subprocess.TimeoutExpired:
            raise PreflightError(
                f"Model init timed out after {_PREFLIGHT_TIMEOUT_SECONDS}s",
                stage="model_init",
            )

    print(f"Preflight OK: {script_path}")
