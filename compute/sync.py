import subprocess
import shlex
from pathlib import Path

_DEFAULT_REMOTE_DIR = "/workspace/parameter-golf-autoresearch/"
_DEFAULT_LOCAL_DIR = "./runpod_results/"
_DEFAULT_NPROC = 8
_DEFAULT_VOCAB_SIZE = 1024
_DEFAULT_TIMEOUT = 1800  # 30 min — 600s train + GPTQ + eval overhead
_TIMEOUT_EXIT_CODE = -1
_RSYNC_BIN = "rsync"
_RSYNC_FLAGS = "-avz"
_SSH_BIN = "ssh"
_PORT_FLAG = "-p"
_OPT_FLAG = "-o"
_STRICT_HOST_CHECK = "StrictHostKeyChecking=no"
_SHELL_FLAG = "-e"


def _ssh_shell_arg(port: int) -> str:
    return f"{_SSH_BIN} {_PORT_FLAG} {port} {_OPT_FLAG} {_STRICT_HOST_CHECK}"


def _build_rsync_cmd(port: int, source: str, destination: str) -> list[str]:
    return [
        _RSYNC_BIN,
        _RSYNC_FLAGS,
        _SHELL_FLAG,
        _ssh_shell_arg(port),
        source,
        destination,
    ]


def _build_ssh_cmd(port: int, user_host: str, remote_command: str) -> list[str]:
    return [
        _SSH_BIN,
        _PORT_FLAG,
        str(port),
        _OPT_FLAG,
        _STRICT_HOST_CHECK,
        user_host,
        remote_command,
    ]


def push_to_pod(
    ssh_conn: str, local_files: list[str], remote_dir: str = _DEFAULT_REMOTE_DIR
) -> None:
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    for local_file in local_files:
        cmd = _build_rsync_cmd(port, local_file, f"{user_host}:{remote_dir}")
        print(f"Pushing {local_file} -> {user_host}:{remote_dir}")
        subprocess.run(cmd, check=True)


def pull_from_pod(
    ssh_conn: str, remote_files: list[str], local_dir: str = _DEFAULT_LOCAL_DIR, optional: bool = False
) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    for remote_file in remote_files:
        cmd = _build_rsync_cmd(port, f"{user_host}:{remote_file}", local_dir)
        print(f"Pulling {user_host}:{remote_file} -> {local_dir}")
        result = subprocess.run(cmd, check=not optional)
        if optional and result.returncode != 0:
            print(f"  (optional file not found, skipping)")


def run_remote_training(
    ssh_conn: str,
    run_id: str,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    nproc: int = _DEFAULT_NPROC,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
    env_vars: dict[str, str] | None = None,
) -> int:
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    # After training, copy artifacts to home (~/) so pull_from_pod can find them via relative paths.
    # run.log, logs/, final_model.int6.zst, and final_model.pt are written to /workspace by torchrun.
    _wd = _DEFAULT_REMOTE_DIR.rstrip("/")
    extra_env = ""
    if env_vars:
        extra_env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars.items()) + " "
    # Use pod's pre-installed full dataset (195 shards / 19.5B tokens) if available.
    # Check shard 000001 (not 000000 which we also push) to confirm it's the full dataset.
    # Tries multiple paths; falls back to the 1-shard data we pushed.
    _data_detect = (
        f"_dp={_wd}/data/datasets/fineweb10B_sp1024; "
        f"for _try in "
        f"/data/datasets/fineweb10B_sp1024 "
        f"/workspace/data/datasets/fineweb10B_sp1024 "
        f"/workspace/datasets/fineweb10B_sp1024 "
        f"/opt/datasets/fineweb10B_sp1024; do "
        f"if ls $_try/fineweb_train_000001.bin 2>/dev/null; then _dp=$_try; break; fi; "
        f"done; "
        f"export DATA_PATH=$_dp; "
        f"echo \"DATA_PATH=$DATA_PATH (train shards: $(ls $DATA_PATH/fineweb_train_*.bin 2>/dev/null | wc -l))\"; "
    )
    train_cmd = (
        f"cd {_wd} && "
        f"_copy_artifacts() {{ "
        f"cp {_wd}/run.log ~/run.log 2>/dev/null; "
        f"cp -r {_wd}/logs ~/logs 2>/dev/null; "
        f"cp {_wd}/final_model.int6.zst ~/model.zst 2>/dev/null; "
        f"cp {_wd}/final_model.pt ~/model.bin 2>/dev/null; "
        f"}}; "
        f"trap _copy_artifacts EXIT; "
        f"pip install -q zstandard 2>/dev/null; "
        f"{_data_detect}"
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
