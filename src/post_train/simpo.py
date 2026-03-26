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


def load_model_and_tokenizer(cfg):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        base_model = base_model.to("cuda")

    if _looks_like_adapter_dir(cfg.model_name_or_path):
        model = PeftModel.from_pretrained(base_model, cfg.model_name_or_path, is_trainable=True)
    elif cfg.model_name_or_path == cfg.base_model_name:
        model = base_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    return model, tokenizer


def train_simpo(cfg) -> Path:
    from trl import CPOConfig, CPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = load_preference_dataset(cfg.train_dataset)
    trainer = CPOTrainer(
        model=model,
        args=CPOConfig(
            output_dir=str(cfg.output_dir),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            report_to=cfg.report_to,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            seed=cfg.seed,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_prompt_length,
            max_completion_length=cfg.max_completion_length,
            loss_type="simpo",
            beta=cfg.beta,
            simpo_gamma=cfg.simpo_gamma,
            cpo_alpha=cfg.cpo_alpha,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
