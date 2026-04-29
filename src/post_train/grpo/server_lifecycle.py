from __future__ import annotations

import atexit
import ctypes
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .gpu import gpu_is_free


def health_check(base_url: str, timeout_seconds: float = 2.0) -> bool:
    health_url = base_url.rstrip("/") + "/health/"
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (URLError, TimeoutError):
        return False


def port_is_open(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def wait_for_vllm_server(base_url: str, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if health_check(base_url):
            return
        time.sleep(2.0)
    raise TimeoutError(f"等待 vLLM server 就绪超时：{base_url.rstrip('/') + '/health/'}")


def read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def is_owned_vllm_process(pid: int) -> bool:
    cmdline = read_cmdline(pid)
    return "trl.cli" in cmdline and "vllm-serve" in cmdline


def load_server_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_server_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def clear_server_metadata(path: Path) -> None:
    if path.exists():
        path.unlink()


def cleanup_server(metadata: dict, metadata_path: Path) -> None:
    server_pid = metadata.get("server_pid")
    if not server_pid:
        clear_server_metadata(metadata_path)
        return
    try:
        pgid = os.getpgid(int(server_pid))
    except ProcessLookupError:
        clear_server_metadata(metadata_path)
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + (10 if sig == signal.SIGTERM else 5)
        while time.time() < deadline:
            try:
                os.kill(int(server_pid), 0)
            except OSError:
                clear_server_metadata(metadata_path)
                return
            time.sleep(0.5)
    clear_server_metadata(metadata_path)


def prepare_vllm_server_child() -> None:
    os.setsid()
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # 父 wrapper 被异常杀掉时，让 vLLM 进程收到 SIGTERM，减少孤儿 server 残留。
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


def install_exit_cleanup(cleanup_func):
    cleanup_done = False

    def run_cleanup_once() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        cleanup_func()

    old_handlers = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        if hasattr(signal, sig.name)
    }

    def handle_signal(signum, frame) -> None:
        run_cleanup_once()
        raise SystemExit(128 + int(signum))

    atexit.register(run_cleanup_once)
    for sig in old_handlers:
        signal.signal(sig, handle_signal)

    def restore() -> None:
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)
        try:
            atexit.unregister(run_cleanup_once)
        except ValueError:
            pass

    return run_cleanup_once, restore


def metadata_matches(
    metadata: dict,
    *,
    base_url: str,
    model_path: str,
    vllm_gpus: list[int],
    trainer_gpu: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    min_free_memory_mb: int,
    max_gpu_utilization: int,
) -> bool:
    if metadata.get("base_url") != base_url:
        return False
    if metadata.get("model_path") != model_path:
        return False
    metadata_vllm_gpus = metadata.get("vllm_gpus", [metadata.get("vllm_gpu")])
    if metadata_vllm_gpus != vllm_gpus:
        return False
    if metadata.get("trainer_gpu") != trainer_gpu:
        return False
    if metadata.get("max_model_len") != max_model_len:
        return False
    if float(metadata.get("gpu_memory_utilization", -1.0)) != float(gpu_memory_utilization):
        return False
    server_pid = metadata.get("server_pid")
    if not server_pid or not is_owned_vllm_process(int(server_pid)):
        return False
    if not gpu_is_free(
        trainer_gpu,
        min_free_memory_mb=min_free_memory_mb,
        max_gpu_utilization=max_gpu_utilization,
    ):
        return False
    return True
