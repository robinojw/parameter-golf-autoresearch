import subprocess
import shlex
import time
from pathlib import Path

from compute.runpod_client import _find_ssh_private_key

_DEFAULT_REMOTE_DIR = "/workspace/parameter-golf-autoresearch/"
_DEFAULT_LOCAL_DIR = "./runpod_results/"
_DEFAULT_NPROC = 8
_DEFAULT_TIMEOUT_SECONDS = 2400
_TIMEOUT_EXIT_CODE = -1
_RSYNC_NOT_FOUND_EXIT_CODE = 127
_RSYNC_BIN = "rsync"
_RSYNC_FLAGS = "-avz"
_SCP_BIN = "scp"
_SSH_BIN = "ssh"
_PORT_FLAG = "-p"
_OPT_FLAG = "-o"
_IDENTITY_FLAG = "-i"
_SHELL_FLAG = "-e"
_STRICT_HOST_CHECK = "StrictHostKeyChecking=no"
_CONNECT_TIMEOUT = "ConnectTimeout=30"
_SERVER_ALIVE = "ServerAliveInterval=15"
_SERVER_ALIVE_MAX = "ServerAliveCountMax=3"
_RSYNC_TIMEOUT_SECONDS = 120
_SCP_TIMEOUT_SECONDS = 120
_SSH_CONNECT_RETRIES = 3
_SSH_RETRY_DELAY_SECONDS = 15


def _ssh_shell_arg(port: int) -> str:
    identity_file = _find_ssh_private_key()
    identity_part = f" {_IDENTITY_FLAG} {identity_file}" if identity_file else ""
    return (
        f"{_SSH_BIN} {_PORT_FLAG} {port}"
        f"{identity_part}"
        f" {_OPT_FLAG} {_STRICT_HOST_CHECK}"
        f" {_OPT_FLAG} {_CONNECT_TIMEOUT}"
        f" {_OPT_FLAG} {_SERVER_ALIVE}"
        f" {_OPT_FLAG} {_SERVER_ALIVE_MAX}"
    )


def _build_rsync_cmd(port: int, source: str, destination: str) -> list[str]:
    return [
        _RSYNC_BIN,
        _RSYNC_FLAGS,
        _SHELL_FLAG,
        _ssh_shell_arg(port),
        source,
        destination,
    ]


def _build_scp_cmd(
    port: int, source: str, destination: str, recursive: bool = False
) -> list[str]:
    """Build an scp command as fallback when rsync is unavailable."""
    cmd = [_SCP_BIN]
    if recursive:
        cmd.append("-r")
    cmd.extend([_PORT_FLAG, str(port)])
    cmd.extend([_OPT_FLAG, _STRICT_HOST_CHECK])
    cmd.extend([_OPT_FLAG, _CONNECT_TIMEOUT])
    identity_file = _find_ssh_private_key()
    if identity_file:
        cmd.extend([_IDENTITY_FLAG, identity_file])
    cmd.extend([source, destination])
    return cmd


def _build_ssh_cmd(port: int, user_host: str, remote_command: str) -> list[str]:
    cmd = [
        _SSH_BIN,
        _PORT_FLAG,
        str(port),
        _OPT_FLAG, _STRICT_HOST_CHECK,
        _OPT_FLAG, _CONNECT_TIMEOUT,
        _OPT_FLAG, _SERVER_ALIVE,
        _OPT_FLAG, _SERVER_ALIVE_MAX,
    ]
    identity_file = _find_ssh_private_key()
    if identity_file:
        cmd.extend([_IDENTITY_FLAG, identity_file])
    cmd.extend([user_host, remote_command])
    return cmd


def _push_single_file_scp(
    port: int, local_file: str, remote_destination: str
) -> None:
    """Push a single file or directory via scp (fallback when rsync unavailable)."""
    is_dir = Path(local_file).is_dir()
    cmd = _build_scp_cmd(port, local_file, remote_destination, recursive=is_dir)
    print(f"  Falling back to scp for {local_file}")
    result = subprocess.run(cmd, timeout=_SCP_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def push_to_pod(
    ssh_conn: str, local_files: list[str], remote_dir: str = _DEFAULT_REMOTE_DIR
) -> None:
    """Push local files to remote pod, with SCP fallback if rsync unavailable."""
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    remote_destination = f"{user_host}:{remote_dir}"

    # Ensure remote directory exists
    mkdir_cmd = _build_ssh_cmd(port, user_host, f"mkdir -p {remote_dir}")
    subprocess.run(mkdir_cmd, timeout=_SCP_TIMEOUT_SECONDS, check=False)

    for local_file in local_files:
        cmd = _build_rsync_cmd(port, local_file, remote_destination)
        pushed = False
        for attempt in range(1, _SSH_CONNECT_RETRIES + 1):
            print(f"Pushing {local_file} -> {user_host}:{remote_dir} (attempt {attempt}/{_SSH_CONNECT_RETRIES})")
            try:
                result = subprocess.run(cmd, timeout=_RSYNC_TIMEOUT_SECONDS)
                if result.returncode == 0:
                    pushed = True
                    break
                if result.returncode == _RSYNC_NOT_FOUND_EXIT_CODE:
                    _push_single_file_scp(port, local_file, remote_destination)
                    pushed = True
                    break
                is_last_attempt = attempt >= _SSH_CONNECT_RETRIES
                if is_last_attempt:
                    raise subprocess.CalledProcessError(result.returncode, cmd)
                print(f"  rsync failed (exit {result.returncode}), retrying in {_SSH_RETRY_DELAY_SECONDS}s...")
                time.sleep(_SSH_RETRY_DELAY_SECONDS)
            except subprocess.TimeoutExpired:
                is_last_attempt = attempt >= _SSH_CONNECT_RETRIES
                if is_last_attempt:
                    raise
                print(f"  rsync timed out after {_RSYNC_TIMEOUT_SECONDS}s, retrying in {_SSH_RETRY_DELAY_SECONDS}s...")
                time.sleep(_SSH_RETRY_DELAY_SECONDS)
        if not pushed:
            raise RuntimeError(f"Failed to push {local_file} after {_SSH_CONNECT_RETRIES} attempts")


def pull_from_pod(
    ssh_conn: str, remote_files: list[str], local_dir: str = _DEFAULT_LOCAL_DIR, optional: bool = False
) -> None:
    """Pull remote files from pod, with SCP fallback and retry logic."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    for remote_file in remote_files:
        remote_source = f"{user_host}:{remote_file}"
        cmd = _build_rsync_cmd(port, remote_source, local_dir)
        pulled = False
        for attempt in range(1, _SSH_CONNECT_RETRIES + 1):
            print(f"Pulling {remote_source} -> {local_dir} (attempt {attempt}/{_SSH_CONNECT_RETRIES})")
            try:
                result = subprocess.run(cmd, timeout=_RSYNC_TIMEOUT_SECONDS)
                if result.returncode == 0:
                    pulled = True
                    break
                if result.returncode == _RSYNC_NOT_FOUND_EXIT_CODE:
                    scp_cmd = _build_scp_cmd(port, remote_source, local_dir)
                    print(f"  Falling back to scp for {remote_file}")
                    scp_result = subprocess.run(scp_cmd, timeout=_SCP_TIMEOUT_SECONDS)
                    if scp_result.returncode == 0:
                        pulled = True
                    break
                is_last_attempt = attempt >= _SSH_CONNECT_RETRIES
                if is_last_attempt:
                    break
                print(f"  rsync failed (exit {result.returncode}), retrying in {_SSH_RETRY_DELAY_SECONDS}s...")
                time.sleep(_SSH_RETRY_DELAY_SECONDS)
            except subprocess.TimeoutExpired:
                is_last_attempt = attempt >= _SSH_CONNECT_RETRIES
                if is_last_attempt:
                    break
                print(f"  rsync timed out, retrying in {_SSH_RETRY_DELAY_SECONDS}s...")
                time.sleep(_SSH_RETRY_DELAY_SECONDS)
        if not pulled and not optional:
            raise RuntimeError(f"Failed to pull {remote_file} after {_SSH_CONNECT_RETRIES} attempts")
        if not pulled and optional:
            print("  (optional file not found or transfer failed, skipping)")


def run_remote_training(
    ssh_conn: str,
    run_id: str,
    nproc: int = _DEFAULT_NPROC,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    env_vars: dict[str, str] | None = None,
) -> int:
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    workspace_dir = _DEFAULT_REMOTE_DIR.rstrip("/")
    extra_env = ""
    if env_vars:
        extra_env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars.items()) + " "
    data_detect_script = (
        f"_dp={workspace_dir}/data/datasets/fineweb10B_sp1024; "
        f"for _try in "
        f"/data/datasets/fineweb10B_sp1024 "
        f"/workspace/data/datasets/fineweb10B_sp1024 "
        f"/workspace/datasets/fineweb10B_sp1024 "
        f"/opt/datasets/fineweb10B_sp1024; do "
        f"if ls $_try/fineweb_train_000001.bin 2>/dev/null; then _dp=$_try; break; fi; "
        f"done; "
        f"if [ \"$_dp\" = \"{workspace_dir}/data/datasets/fineweb10B_sp1024\" ]; then "
        f"echo 'Full dataset not pre-installed, downloading 32 shards from HuggingFace...'; "
        f"cd {workspace_dir}/data && python3 cached_challenge_fineweb.py --train-shards 32 || true; cd {workspace_dir}; "
        f"fi; "
        f"export DATA_PATH=$_dp; "
        f"_nshards=$(ls $DATA_PATH/fineweb_train_*.bin 2>/dev/null | wc -l); "
        f"echo \"DATA_PATH=$DATA_PATH (train shards: $_nshards)\"; "
        f"if [ $_nshards -ge 4 ]; then export LOADER_MODE=coprime; "
        f"else export LOADER_MODE=sequential; fi; "
        f"echo \"LOADER_MODE=$LOADER_MODE\"; "
    )
    train_cmd = (
        f"cd {workspace_dir} && "
        f"_copy_artifacts() {{ "
        f"cp {workspace_dir}/run.log ~/run.log 2>/dev/null; "
        f"cp -r {workspace_dir}/logs ~/logs 2>/dev/null; "
        f"cp {workspace_dir}/final_model.int6.zst ~/model.zst 2>/dev/null; "
        f"cp {workspace_dir}/final_model.pt ~/model.bin 2>/dev/null; "
        f"}}; "
        f"trap _copy_artifacts EXIT; "
        f"python3 -m pip install -q zstandard --break-system-packages 2>&1 | grep -v 'already satisfied' || true; "
        f"{data_detect_script}"
        f"{extra_env}RUN_ID={shlex.quote(run_id)} "
        f"torchrun --standalone --nproc_per_node={nproc} train_gpt.py"
    )
    cmd = _build_ssh_cmd(port, user_host, train_cmd)
    print(f"Running training on {user_host}: {train_cmd}")
    try:
        result = subprocess.run(cmd, timeout=timeout_seconds)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"Training timed out after {timeout_seconds}s, killing remote process")
        kill_cmd = _build_ssh_cmd(port, user_host, "pkill -f train_gpt.py")
        subprocess.run(kill_cmd, check=False)
        return _TIMEOUT_EXIT_CODE


def _parse_ssh_conn(ssh_conn: str) -> tuple[str, str, int]:
    parts = ssh_conn.split()
    user_host = parts[0]
    port = int(parts[parts.index(_PORT_FLAG) + 1])
    host = user_host.split("@")[1]
    return user_host, host, port
