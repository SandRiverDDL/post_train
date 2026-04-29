from __future__ import annotations

import subprocess


def query_gpu_status() -> list[tuple[int, int, int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int, int, int]] = []
    for line in result.stdout.strip().splitlines():
        idx, used, total, gpu_util = [part.strip() for part in line.split(",")]
        rows.append((int(idx), int(used), int(total), int(gpu_util)))
    return rows


def gpu_is_free(gpu_index: int, *, min_free_memory_mb: int, max_gpu_utilization: int) -> bool:
    for idx, used, total, gpu_util in query_gpu_status():
        if idx == gpu_index:
            return (total - used) >= min_free_memory_mb and gpu_util <= max_gpu_utilization
    raise RuntimeError(f"未找到 GPU {gpu_index}。")


def validate_gpu_layout(*, trainer_gpu: int | None, vllm_gpus: list[int] | None) -> None:
    if (trainer_gpu is None) != (vllm_gpus is None):
        raise ValueError("trainer_gpu 与 vllm_gpus 必须同时指定或同时留空。")
    if not vllm_gpus:
        return
    if len(set(vllm_gpus)) != len(vllm_gpus):
        raise ValueError("vllm_gpus 不能包含重复 GPU。")
    if trainer_gpu in vllm_gpus:
        raise ValueError("trainer_gpu 不能和 vLLM GPU 重叠。")


def select_grpo_gpu_pair(
    *,
    preferred_trainer_gpu: int = 6,
    preferred_vllm_gpu: int = 7,
    min_free_memory_mb: int = 20 * 1024,
    max_gpu_utilization: int = 10,
) -> tuple[int, int]:
    gpu_rows = query_gpu_status()
    free_gpus = {
        idx
        for idx, used, total, gpu_util in gpu_rows
        if (total - used) >= min_free_memory_mb and gpu_util <= max_gpu_utilization
    }
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
    usage = ", ".join(f"gpu{idx}={used}/{total}MB util={gpu_util}%" for idx, used, total, gpu_util in gpu_rows)
    raise RuntimeError(f"未找到两张空闲 GPU 可用于 GRPO 训练与 vLLM server：{usage}")
