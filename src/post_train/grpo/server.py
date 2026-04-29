from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from post_train.config import load_grpo_train_config

from .gpu import gpu_is_free, select_grpo_gpu_pair, validate_gpu_layout
from .runtime import GRPO_VLLM_SERVER_BASE_URL_ENV
from .server_lifecycle import (
    cleanup_server,
    health_check,
    install_exit_cleanup,
    load_server_metadata,
    metadata_matches,
    port_is_open,
    prepare_vllm_server_child,
    wait_for_vllm_server,
    write_server_metadata,
)


def run_grpo_with_vllm_server(config_path: str | Path) -> int:
    return run_grpo_with_vllm_server_on_gpus(config_path)


def _resolve_vllm_gpus(*, vllm_gpu: int | None, vllm_gpus: list[int] | None) -> list[int] | None:
    if vllm_gpu is not None and vllm_gpus is not None:
        raise ValueError("vllm_gpu 与 vllm_gpus 只能指定一个。")
    return [vllm_gpu] if vllm_gpu is not None else vllm_gpus


def _server_metadata_payload(
    *,
    base_url: str,
    port: int,
    model_path: str,
    trainer_gpu: int,
    vllm_gpus: list[int],
    max_model_len: int,
    gpu_memory_utilization: float,
    server_pid: int,
    server_log_path: Path,
    status: str,
) -> dict:
    return {
        "base_url": base_url,
        "port": port,
        "model_path": model_path,
        "trainer_gpu": trainer_gpu,
        "vllm_gpu": vllm_gpus[0],
        "vllm_gpus": vllm_gpus,
        "vllm_data_parallel_size": len(vllm_gpus),
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "server_pid": server_pid,
        "server_log": str(server_log_path),
        "started_at": time.time(),
        "status": status,
    }


def run_grpo_with_vllm_server_on_gpus(
    config_path: str | Path,
    *,
    trainer_gpu: int | None = None,
    vllm_gpu: int | None = None,
    vllm_gpus: list[int] | None = None,
    vllm_port: int | None = None,
) -> int:
    resolved_vllm_gpus = _resolve_vllm_gpus(vllm_gpu=vllm_gpu, vllm_gpus=vllm_gpus)
    validate_gpu_layout(trainer_gpu=trainer_gpu, vllm_gpus=resolved_vllm_gpus)

    cfg = load_grpo_train_config(config_path)
    if not cfg.use_vllm or cfg.vllm_mode != "server":
        raise ValueError("该入口只支持 use_vllm=true 且 vllm_mode=server 的配置。")

    base_url = cfg.vllm_server_base_url or f"http://{cfg.vllm_server_host}:{cfg.vllm_server_port}"
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = vllm_port if vllm_port is not None else (parsed.port or cfg.vllm_server_port or 8000)
    base_url = f"{parsed.scheme or 'http'}://{host}:{port}"
    max_model_len = cfg.max_prompt_length + cfg.max_completion_length
    min_free_memory_mb = 20 * 1024
    max_gpu_utilization = 10

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "vllm_server.log"
    metadata_path = Path(".runtime/grpo_vllm_server.json")

    metadata = load_server_metadata(metadata_path)
    reused_server = False
    server_started_by_this_run = False
    server_proc: subprocess.Popen | None = None
    cleanup_once = None
    restore_exit_cleanup = None
    selected_trainer_gpu: int
    selected_vllm_gpus: list[int]

    if health_check(base_url):
        if metadata is None:
            raise RuntimeError(
                f"检测到已有可用 vLLM server ({base_url})，但未找到 metadata：{metadata_path}。请先手动关闭旧 server。"
            )
        metadata_trainer_gpu = int(metadata.get("trainer_gpu", -1))
        metadata_vllm_gpus = metadata.get("vllm_gpus", [metadata.get("vllm_gpu", -1)])
        candidate_trainer_gpu = trainer_gpu if trainer_gpu is not None else metadata_trainer_gpu
        candidate_vllm_gpus = resolved_vllm_gpus if resolved_vllm_gpus is not None else metadata_vllm_gpus
        if metadata_matches(
            metadata,
            base_url=base_url,
            model_path=cfg.base_model_name,
            vllm_gpus=candidate_vllm_gpus,
            trainer_gpu=candidate_trainer_gpu,
            max_model_len=max_model_len,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            min_free_memory_mb=min_free_memory_mb,
            max_gpu_utilization=max_gpu_utilization,
        ):
            reused_server = True
            selected_trainer_gpu = candidate_trainer_gpu
            selected_vllm_gpus = candidate_vllm_gpus
        else:
            cleanup_server(metadata, metadata_path)
            metadata = None
            if health_check(base_url):
                raise RuntimeError(
                    f"检测到已有 vLLM server ({base_url})，但 metadata 不匹配且未能安全清理。"
                    "请先确认该 server 是否属于当前任务，再手动关闭。"
                )
    elif port_is_open(host, port):
        raise RuntimeError(
            f"检测到 {host}:{port} 已被占用，但 vLLM health check 不可用。"
            "这通常是旧 vLLM server 残留或卡死。请先确认占用进程属于当前任务，再手动关闭。"
        )

    if not reused_server:
        if trainer_gpu is None or resolved_vllm_gpus is None:
            selected_trainer_gpu, selected_vllm_gpu = select_grpo_gpu_pair(
                min_free_memory_mb=min_free_memory_mb,
                max_gpu_utilization=max_gpu_utilization,
            )
            selected_vllm_gpus = [selected_vllm_gpu]
        else:
            if not gpu_is_free(
                trainer_gpu,
                min_free_memory_mb=min_free_memory_mb,
                max_gpu_utilization=max_gpu_utilization,
            ):
                raise RuntimeError(f"指定 trainer GPU {trainer_gpu} 空闲显存不足或 GPU 利用率过高。")
            for gpu_index in resolved_vllm_gpus:
                if not gpu_is_free(
                    gpu_index,
                    min_free_memory_mb=min_free_memory_mb,
                    max_gpu_utilization=max_gpu_utilization,
                ):
                    raise RuntimeError(f"指定 vLLM GPU {gpu_index} 空闲显存不足或 GPU 利用率过高。")
            selected_trainer_gpu, selected_vllm_gpus = trainer_gpu, resolved_vllm_gpus

        trainer_gpu, vllm_gpus = selected_trainer_gpu, selected_vllm_gpus
        print(f"grpo.selected_trainer_gpu={trainer_gpu}")
        print(f"grpo.selected_vllm_gpus={','.join(str(gpu) for gpu in vllm_gpus)}")
        print(f"grpo.vllm_server_base_url={base_url}")
        print(f"grpo.vllm_server_log={server_log_path}")

        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in vllm_gpus)
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
            str(cfg.vllm_gpu_memory_utilization),
            "--max_model_len",
            str(max_model_len),
            "--data_parallel_size",
            str(len(vllm_gpus)),
            "--trust_remote_code",
        ]
        server_log = server_log_path.open("w")
        server_proc = subprocess.Popen(
            server_cmd,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=prepare_vllm_server_child,
        )
        write_server_metadata(
            metadata_path,
            _server_metadata_payload(
                base_url=base_url,
                port=port,
                model_path=cfg.base_model_name,
                trainer_gpu=trainer_gpu,
                vllm_gpus=vllm_gpus,
                max_model_len=max_model_len,
                gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                server_pid=server_proc.pid,
                server_log_path=server_log_path,
                status="starting",
            ),
        )
        server_started_by_this_run = True

        def cleanup_this_run_server() -> None:
            latest_metadata = load_server_metadata(metadata_path)
            if latest_metadata is not None:
                cleanup_server(latest_metadata, metadata_path)

        cleanup_once, restore_exit_cleanup = install_exit_cleanup(cleanup_this_run_server)
        try:
            wait_for_vllm_server(base_url)
        except BaseException:
            cleanup_once()
            raise
        write_server_metadata(
            metadata_path,
            _server_metadata_payload(
                base_url=base_url,
                port=port,
                model_path=cfg.base_model_name,
                trainer_gpu=trainer_gpu,
                vllm_gpus=vllm_gpus,
                max_model_len=max_model_len,
                gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                server_pid=server_proc.pid,
                server_log_path=server_log_path,
                status="ready",
            ),
        )
    else:
        trainer_gpu = int(metadata["trainer_gpu"])
        metadata_vllm_gpus = metadata.get("vllm_gpus")
        if metadata_vllm_gpus is None:
            metadata_vllm_gpus = [metadata["vllm_gpu"]]
        vllm_gpus = [int(gpu) for gpu in metadata_vllm_gpus]
        print("grpo.reusing_vllm_server=true")
        print(f"grpo.selected_trainer_gpu={trainer_gpu}")
        print(f"grpo.selected_vllm_gpus={','.join(str(gpu) for gpu in vllm_gpus)}")
        print(f"grpo.vllm_server_base_url={base_url}")
        print(f"grpo.vllm_server_log={metadata.get('server_log', '')}")

    train_cmd = [sys.executable, "scripts/train_grpo.py", "--config", str(config_path)]
    trainer_env = os.environ.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = str(trainer_gpu)
    trainer_env[GRPO_VLLM_SERVER_BASE_URL_ENV] = base_url
    try:
        return subprocess.run(train_cmd, env=trainer_env, check=False).returncode
    finally:
        if server_started_by_this_run:
            if cleanup_once is not None:
                cleanup_once()
            if restore_exit_cleanup is not None:
                restore_exit_cleanup()
