from __future__ import annotations

from typing import Any

from .model_loading import looks_like_adapter_dir


def parameter_count(model, *, trainable_only: bool = False, name_substring: str | None = None) -> int:
    total = 0
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if name_substring is not None and name_substring not in name:
            continue
        total += int(parameter.numel())
    return total


def collect_adapter_diagnostics(model, cfg) -> dict[str, Any]:
    requested_path = str(cfg.model_name_or_path)
    requested_is_adapter_dir = looks_like_adapter_dir(requested_path)
    is_peft_model = hasattr(model, "peft_config")
    lora_param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    diagnostics: dict[str, Any] = {
        "base_model_name": str(cfg.base_model_name),
        "requested_model_name_or_path": requested_path,
        "requested_path_is_adapter_dir": requested_is_adapter_dir,
        "is_peft_model": bool(is_peft_model),
        "trainable_param_count": parameter_count(model, trainable_only=True),
        "total_param_count": parameter_count(model),
        "lora_param_count": parameter_count(model, name_substring="lora_"),
        "trainable_lora_param_count": parameter_count(model, trainable_only=True, name_substring="lora_"),
        "lora_param_examples": lora_param_names[:4],
    }

    active_adapter = None
    if hasattr(model, "active_adapter"):
        active_adapter = getattr(model, "active_adapter")
    elif hasattr(model, "active_adapters"):
        try:
            active_adapter = model.active_adapters()
        except TypeError:
            active_adapter = model.active_adapters
    diagnostics["active_adapter"] = active_adapter

    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        diagnostics["available_adapters"] = list(peft_config.keys())
        resolved_adapter_name = None
        if isinstance(active_adapter, str) and active_adapter in peft_config:
            resolved_adapter_name = active_adapter
        elif isinstance(active_adapter, (list, tuple)) and active_adapter:
            candidate = active_adapter[0]
            if candidate in peft_config:
                resolved_adapter_name = candidate
        if resolved_adapter_name is None:
            resolved_adapter_name = next(iter(peft_config))
        adapter_cfg = peft_config[resolved_adapter_name]
        diagnostics["resolved_adapter_name"] = resolved_adapter_name
        diagnostics["adapter_base_model_name_or_path"] = getattr(adapter_cfg, "base_model_name_or_path", None)

    if requested_is_adapter_dir and not is_peft_model:
        raise RuntimeError(f"GRPO 期望从 adapter checkpoint 继续训练，但当前模型不是 PeftModel：{requested_path}")
    if requested_is_adapter_dir and diagnostics["lora_param_count"] <= 0:
        raise RuntimeError(f"GRPO 期望从 adapter checkpoint 继续训练，但未检测到任何 LoRA 参数：{requested_path}")
    return diagnostics
