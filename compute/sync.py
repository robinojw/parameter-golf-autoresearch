import subprocess
import shlex
from pathlib import Path

_DEFAULT_REMOTE_DIR = "/workspace/parameter-golf-autoresearch/"
_DEFAULT_LOCAL_DIR = "./runpod_results/"
_DEFAULT_NPROC = 8
_DEFAULT_VOCAB_SIZE = 1024
_DEFAULT_TIMEOUT = 720
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
    ssh_conn: str, remote_files: list[str], local_dir: str = _DEFAULT_LOCAL_DIR
) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    for remote_file in remote_files:
        cmd = _build_rsync_cmd(port, f"{user_host}:{remote_file}", local_dir)
        print(f"Pulling {user_host}:{remote_file} -> {local_dir}")
        subprocess.run(cmd, check=True)


def run_remote_training(
    ssh_conn: str,
    run_id: str,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    nproc: int = _DEFAULT_NPROC,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
) -> int:
    user_host, _host, port = _parse_ssh_conn(ssh_conn)
    train_cmd = (
        f"cd {_DEFAULT_REMOTE_DIR} && "
        f"RUN_ID={shlex.quote(run_id)} "
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
