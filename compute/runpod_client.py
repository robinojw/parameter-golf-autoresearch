import runpod
import atexit
import signal
import time

_DEFAULT_GPU_COUNT = 8
_DEFAULT_VOLUME_GB = 50
_DEFAULT_TIMEOUT = 300
_SSH_PRIVATE_PORT = 22
_POLL_INTERVAL = 5
_PUBLIC_IP_KEY = "publicIp"


class RunPodClient:
    def __init__(self, api_key: str, template_id: str) -> None:
        self.api_key = api_key
        self.template_id = template_id
        self._active_pods: set[str] = set()
        runpod.api_key = api_key
        atexit.register(self._cleanup_all)
        signal.signal(signal.SIGTERM, lambda signum, frame: self._cleanup_all())

    def launch_pod(
        self,
        gpu_count: int = _DEFAULT_GPU_COUNT,
        gpu_type: str = "NVIDIA H100 SXM5 80GB",
    ) -> str:
        pod = runpod.create_pod(
            name=f"pgolf-{int(time.time())}",
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id=gpu_type,
            gpu_count=gpu_count,
            template_id=self.template_id,
            volume_in_gb=_DEFAULT_VOLUME_GB,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        self._active_pods.add(pod_id)
        print(f"Launched pod {pod_id} ({gpu_count}x {gpu_type})")
        return pod_id

    def wait_for_ready(
        self, pod_id: str, timeout_seconds: int = _DEFAULT_TIMEOUT
    ) -> str:
        start = time.time()
        while time.time() - start < timeout_seconds:
            pod = runpod.get_pod(pod_id)
            ssh_conn = self._extract_ssh_conn(pod)
            if ssh_conn:
                print(f"Pod {pod_id} ready: {ssh_conn}")
                return ssh_conn
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout_seconds}s")

    def _extract_ssh_conn(self, pod: dict) -> str | None:
        is_running = pod.get("desiredStatus") == "RUNNING"
        runtime = pod.get("runtime")
        pod_is_usable = is_running and runtime
        if not pod_is_usable:
            return None
        ssh_port = self._find_ssh_port(runtime)
        if not ssh_port:
            return None
        public_ip = self._find_public_ip(runtime)
        if not public_ip:
            return None
        return f"root@{public_ip} -p {ssh_port}"

    def _find_ssh_port(self, runtime: dict) -> int | None:
        ports = runtime.get("ports", [])
        for p in ports:
            if p.get("privatePort") == _SSH_PRIVATE_PORT:
                return p.get("publicPort")
        return None

    def _find_public_ip(self, runtime: dict) -> str | None:
        ip = runtime.get(_PUBLIC_IP_KEY, "")
        if ip:
            return ip
        gpus = runtime.get("gpus", [])
        if gpus:
            return gpus[0].get(_PUBLIC_IP_KEY, "") or None
        for key in ("ip", "host"):
            candidate = runtime.get(key, "")
            if candidate:
                return candidate
        return None

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)
        self._active_pods.discard(pod_id)
        print(f"Terminated pod {pod_id}")

    def get_pod_status(self, pod_id: str) -> dict:
        return runpod.get_pod(pod_id)

    def _cleanup_all(self) -> None:
        if not self._active_pods:
            return
        print(f"Cleaning up {len(self._active_pods)} active pod(s)...")
        for pod_id in list(self._active_pods):
            try:
                runpod.terminate_pod(pod_id)
                print(f"  Terminated {pod_id}")
            except Exception as e:
                print(f"  Failed to terminate {pod_id}: {e}")
        self._active_pods.clear()
