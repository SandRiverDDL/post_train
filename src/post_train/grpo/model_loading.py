from __future__ import annotations

from pathlib import Path
from typing import Any


def looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def resolve_compute_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_train_compute_dtype(cfg):
    return resolve_compute_dtype(cfg.compute_dtype)


def build_quantization_kwargs(cfg) -> dict[str, Any]:
    if not cfg.load_in_4bit:
        return {}

    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        "device_map": "auto",
    }


def aligned_compute_dtype(cfg):
    return resolve_train_compute_dtype(cfg)


def candidate_model_nodes(model) -> list[tuple[str, Any]]:
    seen_ids: set[int] = set()
    queue: list[tuple[str, Any]] = [("model", model)]
    candidates: list[tuple[str, Any]] = []

    while queue:
        path, node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        candidates.append((path, node))

        if hasattr(node, "base_model"):
            queue.append((f"{path}.base_model", getattr(node, "base_model")))
        if hasattr(node, "model"):
            queue.append((f"{path}.model", getattr(node, "model")))
        if hasattr(node, "get_base_model"):
            try:
                queue.append((f"{path}.get_base_model()", node.get_base_model()))
            except Exception:
                pass

    return candidates


def iter_output_heads(model) -> list[tuple[str, Any]]:
    seen_ids: set[int] = set()
    heads: list[tuple[str, Any]] = []

    for path, node in candidate_model_nodes(model):
        lm_head = getattr(node, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight") and id(lm_head) not in seen_ids:
            seen_ids.add(id(lm_head))
            heads.append((f"{path}.lm_head", lm_head))

        if hasattr(node, "get_output_embeddings"):
            try:
                output_embeddings = node.get_output_embeddings()
            except Exception:
                output_embeddings = None
            if output_embeddings is not None and hasattr(output_embeddings, "weight") and id(output_embeddings) not in seen_ids:
                seen_ids.add(id(output_embeddings))
                heads.append((f"{path}.get_output_embeddings()", output_embeddings))

    return heads


def align_output_head_dtypes(model, *, target_dtype) -> list[str]:
    touched_paths: list[str] = []
    for path, head in iter_output_heads(model):
        if head.weight.dtype != target_dtype:
            head.weight.data = head.weight.data.to(target_dtype)
        touched_paths.append(path)
    return touched_paths


def collect_model_dtype_report(model) -> dict[str, str]:
    report: dict[str, str] = {}
    for path, head in iter_output_heads(model):
        report[f"{path}.weight_dtype"] = str(head.weight.dtype)
    if hasattr(model, "get_input_embeddings"):
        try:
            input_embeddings = model.get_input_embeddings()
        except Exception:
            input_embeddings = None
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            report["model.get_input_embeddings().weight_dtype"] = str(input_embeddings.weight.dtype)
    return report


def assert_generation_dtype_ready(model, *, target_dtype) -> dict[str, str]:
    report = collect_model_dtype_report(model)
    expected_dtype = str(target_dtype)
    output_head_items = {key: value for key, value in report.items() if key.endswith(".weight_dtype") and "input_embeddings" not in key}
    if not output_head_items:
        raise RuntimeError("GRPO 生成前 dtype 检查失败：未找到任何可用于生成的 output head。")
    mismatched = {key: value for key, value in output_head_items.items() if value != expected_dtype}
    if mismatched:
        mismatch_text = ", ".join(f"{key}={value}" for key, value in sorted(mismatched.items()))
        raise RuntimeError(f"GRPO 生成前 dtype 检查失败：{mismatch_text}，expected={expected_dtype}。")
    return report


def load_model_and_tokenizer(cfg):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compute_dtype = resolve_train_compute_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    if cfg.load_in_4bit:
        model_kwargs.update(build_quantization_kwargs(cfg))
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name, **model_kwargs)
    if cfg.load_in_4bit and getattr(base_model, "is_loaded_in_4bit", False):
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        base_model = base_model.to("cuda")

    if looks_like_adapter_dir(cfg.model_name_or_path):
        model = PeftModel.from_pretrained(
            base_model,
            cfg.model_name_or_path,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
    elif cfg.model_name_or_path == cfg.base_model_name:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        if cfg.load_in_4bit and getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif torch.cuda.is_available():
            model = model.to("cuda")
    align_output_head_dtypes(model, target_dtype=compute_dtype)
    return model, tokenizer
