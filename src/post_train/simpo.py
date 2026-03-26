from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from post_train.io import ensure_parent, read_jsonl
from post_train.schemas import PreferenceRecord


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def load_preference_dataset(path: str | Path) -> Dataset:
    rows = [PreferenceRecord.model_validate(row).model_dump() for row in read_jsonl(path)]
    return Dataset.from_list(rows)


def _resolve_compute_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_kwargs(cfg) -> dict:
    if not cfg.load_in_4bit:
        return {}

    from transformers import BitsAndBytesConfig

    compute_dtype = _resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        "device_map": "auto",
    }


def build_training_args(cfg):
    import torch
    from trl import CPOConfig

    compute_dtype = _resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)
    use_bf16 = bool(torch.cuda.is_available() and compute_dtype == torch.bfloat16)
    use_fp16 = bool(torch.cuda.is_available() and compute_dtype == torch.float16)

    return CPOConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        bf16=use_bf16,
        fp16=use_fp16,
        loss_type="simpo",
        beta=cfg.beta,
        simpo_gamma=cfg.simpo_gamma,
        cpo_alpha=cfg.cpo_alpha,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
    )


def load_model_and_tokenizer(cfg):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    model_kwargs.update(build_quantization_kwargs(cfg))
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        **model_kwargs,
    )
    if getattr(base_model, "is_loaded_in_4bit", False):
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        base_model = base_model.to("cuda")

    if _looks_like_adapter_dir(cfg.model_name_or_path):
        model = PeftModel.from_pretrained(base_model, cfg.model_name_or_path, is_trainable=True)
    elif cfg.model_name_or_path == cfg.base_model_name:
        model = base_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            **model_kwargs,
        )
        if getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif torch.cuda.is_available():
            model = model.to("cuda")

    return model, tokenizer


def log_training_setup(cfg, model) -> None:
    quantized = bool(getattr(model, "is_loaded_in_4bit", False))
    print(f"simpo.model={cfg.model_name_or_path}")
    print(f"simpo.base_model={cfg.base_model_name}")
    print(f"simpo.load_in_4bit={quantized}")
    print(
        "simpo.lengths="
        f"max={cfg.max_length},prompt={cfg.max_prompt_length},completion={cfg.max_completion_length}"
    )
    print(
        "simpo.batch="
        f"per_device={cfg.batch_size},grad_accum={cfg.gradient_accumulation_steps},logging_steps={cfg.logging_steps}"
    )
    print(
        "simpo.save="
        f"strategy={cfg.save_strategy},steps={cfg.save_steps},limit={cfg.save_total_limit},resume={cfg.resume_from_checkpoint}"
    )


def collect_cuda_memory_stats() -> dict[str, int]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def train_simpo(cfg) -> Path:
    from trl import CPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    log_training_setup(cfg, model)
    train_dataset = load_preference_dataset(cfg.train_dataset)
    training_args = build_training_args(cfg)
    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    memory_stats = collect_cuda_memory_stats()
    if memory_stats:
        print(
            "simpo.cuda_memory="
            f"allocated={memory_stats['max_memory_allocated']},reserved={memory_stats['max_memory_reserved']}"
        )
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
