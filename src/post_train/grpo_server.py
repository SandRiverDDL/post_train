from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from post_train.config import load_grpo_train_config


def _query_gpu_memory() -> list[tuple[int, int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int, int]] = []
    for line in result.stdout.strip().splitlines():
        idx, used, total = [part.strip() for part in line.split(",")]
        rows.append((int(idx), int(used), int(total)))
    return rows


def _gpu_is_free(gpu_index: int, *, min_free_memory_mb: int) -> bool:
    for idx, used, total in _query_gpu_memory():
        if idx == gpu_index:
            return (total - used) >= min_free_memory_mb
    raise RuntimeError(f"未找到 GPU {gpu_index}。")


def select_grpo_gpu_pair(
    *,
    preferred_trainer_gpu: int = 6,
    preferred_vllm_gpu: int = 7,
    min_free_memory_mb: int = 20 * 1024,
) -> tuple[int, int]:
    gpu_rows = _query_gpu_memory()
    free_gpus = {idx for idx, used, total in gpu_rows if (total - used) >= min_free_memory_mb}
    candidate_pairs = [
        (preferred_trainer_gpu, preferred_vllm_gpu),
        (preferred_trainer_gpu, 4),
        (preferred_trainer_gpu, 3),
        (preferred_trainer_gpu, 5),
        (4, preferred_vllm_gpu),
        (4, 6),
        (3, preferred_vllm_gpu),
        (3, 6),
        (4, 3),
        (3, 4),
        (5, 6),
        (6, 5),
        (4, 5),
        (5, 4),
    ]
    for trainer_gpu, vllm_gpu in candidate_pairs:
        if trainer_gpu == vllm_gpu:
            continue
        if trainer_gpu in free_gpus and vllm_gpu in free_gpus:
            return trainer_gpu, vllm_gpu
    usage = ", ".join(f"gpu{idx}={used}/{total}MB" for idx, used, total in gpu_rows)
    raise RuntimeError(f"未找到两张空闲 GPU 可用于 GRPO 训练与 vLLM server：{usage}")


def _health_check(base_url: str, timeout_seconds: float = 2.0) -> bool:
    health_url = base_url.rstrip("/") + "/health/"
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (URLError, TimeoutError):
        return False


def _wait_for_vllm_server(base_url: str, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _health_check(base_url):
            return
        time.sleep(2.0)
    raise TimeoutError(f"等待 vLLM server 就绪超时：{base_url.rstrip('/') + '/health/'}")


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _is_owned_vllm_process(pid: int) -> bool:
    cmdline = _read_cmdline(pid)
    return "trl.cli" in cmdline and "vllm-serve" in cmdline


def _load_server_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_server_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _clear_server_metadata(path: Path) -> None:
    if path.exists():
        path.unlink()


def _cleanup_server(metadata: dict, metadata_path: Path) -> None:
    server_pid = metadata.get("server_pid")
    if not server_pid:
        _clear_server_metadata(metadata_path)
        return
    try:
        pgid = os.getpgid(int(server_pid))
    except ProcessLookupError:
        _clear_server_metadata(metadata_path)
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
                _clear_server_metadata(metadata_path)
                return
            time.sleep(0.5)
    _clear_server_metadata(metadata_path)


def _metadata_matches(
    metadata: dict,
    *,
    base_url: str,
    model_path: str,
    vllm_gpu: int,
    trainer_gpu: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    min_free_memory_mb: int,
) -> bool:
    if metadata.get("base_url") != base_url:
        return False
    if metadata.get("model_path") != model_path:
        return False
    if metadata.get("vllm_gpu") != vllm_gpu:
        return False
    if metadata.get("trainer_gpu") != trainer_gpu:
        return False
    if metadata.get("max_model_len") != max_model_len:
        return False
    if float(metadata.get("gpu_memory_utilization", -1.0)) != float(gpu_memory_utilization):
        return False
    server_pid = metadata.get("server_pid")
    if not server_pid or not _is_owned_vllm_process(int(server_pid)):
        return False
    if not _gpu_is_free(trainer_gpu, min_free_memory_mb=min_free_memory_mb):
        return False
    return True


def run_grpo_with_vllm_server(config_path: str | Path) -> int:
    cfg = load_grpo_train_config(config_path)
    if not cfg.use_vllm or cfg.vllm_mode != "server":
        raise ValueError("该入口只支持 use_vllm=true 且 vllm_mode=server 的配置。")

    base_url = cfg.vllm_server_base_url or f"http://{cfg.vllm_server_host}:{cfg.vllm_server_port}"
    parsed = urlparse(base_url)
    port = parsed.port or 8000
    host = parsed.hostname or "127.0.0.1"
    max_model_len = cfg.max_prompt_length + cfg.max_completion_length
    vllm_gpu_memory_utilization = 0.6
    min_free_memory_mb = 20 * 1024

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "vllm_server.log"
    metadata_path = Path(".runtime/grpo_vllm_server.json")

    selected_trainer_gpu, selected_vllm_gpu = select_grpo_gpu_pair(min_free_memory_mb=min_free_memory_mb)
    metadata = _load_server_metadata(metadata_path)
    reused_server = False
    server_started_by_this_run = False
    server_proc: subprocess.Popen | None = None

    if _health_check(base_url):
        if metadata is None:
            raise RuntimeError(
                f"检测到已有可用 vLLM server ({base_url})，但未找到 metadata：{metadata_path}。请先手动关闭旧 server。"
            )
        if _metadata_matches(
            metadata,
            base_url=base_url,
            model_path=cfg.base_model_name,
            vllm_gpu=selected_vllm_gpu,
            trainer_gpu=selected_trainer_gpu,
            max_model_len=max_model_len,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            min_free_memory_mb=min_free_memory_mb,
        ):
            reused_server = True
        else:
            _cleanup_server(metadata, metadata_path)
            metadata = None

    if not reused_server:
        trainer_gpu, vllm_gpu = selected_trainer_gpu, selected_vllm_gpu
        print(f"grpo.selected_trainer_gpu={trainer_gpu}")
        print(f"grpo.selected_vllm_gpu={vllm_gpu}")
        print(f"grpo.vllm_server_base_url={base_url}")
        print(f"grpo.vllm_server_log={server_log_path}")

        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu)
        server_cmd = [
            sys.executable,
            "-m",
            "trl.cli",
            "vllm-serve",
            "--model",
            cfg.base_model_name,
            "--host",
            host,
            "--port",
            str(port),
            "--gpu_memory_utilization",
            str(vllm_gpu_memory_utilization),
            "--max_model_len",
            str(max_model_len),
            "--trust_remote_code",
        ]
        server_log = server_log_path.open("w")
        server_proc = subprocess.Popen(
            server_cmd,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        _write_server_metadata(
            metadata_path,
            {
                "base_url": base_url,
                "port": port,
                "model_path": cfg.base_model_name,
                "trainer_gpu": trainer_gpu,
                "vllm_gpu": vllm_gpu,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": vllm_gpu_memory_utilization,
                "server_pid": server_proc.pid,
                "server_log": str(server_log_path),
                "started_at": time.time(),
                "status": "starting",
            },
        )
        server_started_by_this_run = True
        try:
            _wait_for_vllm_server(base_url)
        except BaseException:
            latest_metadata = _load_server_metadata(metadata_path)
            if latest_metadata is not None:
                _cleanup_server(latest_metadata, metadata_path)
            raise
        _write_server_metadata(
            metadata_path,
            {
                "base_url": base_url,
                "port": port,
                "model_path": cfg.base_model_name,
                "trainer_gpu": trainer_gpu,
                "vllm_gpu": vllm_gpu,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": vllm_gpu_memory_utilization,
                "server_pid": server_proc.pid,
                "server_log": str(server_log_path),
                "started_at": time.time(),
                "status": "ready",
            },
        )
    else:
        trainer_gpu = int(metadata["trainer_gpu"])
        vllm_gpu = int(metadata["vllm_gpu"])
        print(f"grpo.reusing_vllm_server=true")
        print(f"grpo.selected_trainer_gpu={trainer_gpu}")
        print(f"grpo.selected_vllm_gpu={vllm_gpu}")
        print(f"grpo.vllm_server_base_url={base_url}")
        print(f"grpo.vllm_server_log={metadata.get('server_log', '')}")

    train_cmd = [sys.executable, "scripts/train_grpo.py", "--config", str(config_path)]
    trainer_env = os.environ.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = str(trainer_gpu)
    returncode = 1
    try:
        returncode = subprocess.run(train_cmd, env=trainer_env, check=False).returncode
        return returncode
    finally:
        if server_started_by_this_run and returncode == 0:
            latest_metadata = _load_server_metadata(metadata_path)
            if latest_metadata is not None:
                _cleanup_server(latest_metadata, metadata_path)
