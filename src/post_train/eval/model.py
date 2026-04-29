from __future__ import annotations

from pathlib import Path
from typing import Any


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def _validate_adapter_dir(path: str | Path) -> Path:
    candidate = Path(path)
    missing = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (candidate / name).exists()]
    if missing:
        raise FileNotFoundError(f"adapter 目录缺少必要文件：{candidate}，missing={missing}")
    return candidate


def describe_model_resolution(requested_model: str | None, base_model: str, model_args: dict[str, Any]) -> dict[str, Any]:
    lora_local_path = model_args.get("lora_local_path")
    return {
        "requested_model": requested_model or base_model,
        "base_model": base_model,
        "target_is_adapter_dir": bool(lora_local_path),
        "pretrained": str(model_args["pretrained"]),
        "enable_lora": bool(lora_local_path),
        "lora_local_path": str(lora_local_path) if lora_local_path is not None else None,
        "max_lora_rank": model_args.get("max_lora_rank"),
    }


def resolve_model_args(
    requested_model: str | None,
    base_model: str,
    *,
    backend: str,
    max_length: int,
    device: str,
    attn_implementation: str,
    gpu_memory_utilization: float,
    max_lora_rank: int | None = None,
) -> dict[str, Any]:
    if backend != "vllm":
        raise ValueError("当前评测只支持 backend=vllm。")
    target = requested_model or base_model
    target_is_adapter_dir = _looks_like_adapter_dir(target)
    if target_is_adapter_dir:
        _validate_adapter_dir(target)
    model_args: dict[str, Any] = {
        "pretrained": base_model if target_is_adapter_dir else target,
        "dtype": "auto",
        "trust_remote_code": True,
        "max_length": max_length,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if target_is_adapter_dir:
        model_args["lora_local_path"] = str(target)
        if max_lora_rank is not None:
            model_args["max_lora_rank"] = max_lora_rank
    return model_args
