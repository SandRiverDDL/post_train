from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_train.eval.model import describe_model_resolution, resolve_model_args
from post_train.eval.tasks import build_task_name
from post_train.io import read_jsonl
from post_train.prompts import build_math_prompt, render_chat_prompt


@dataclass(frozen=True)
class VLLMRunnerSpec:
    model_name: str
    llm_kwargs: dict[str, Any]
    max_lora_rank: int | None


def build_vllm_runner_spec(model_args: dict[str, Any]) -> tuple[VLLMRunnerSpec, str | None]:
    llm_kwargs = dict(model_args)
    model_name = str(llm_kwargs.pop("pretrained"))
    max_length = llm_kwargs.pop("max_length", None)
    lora_path = llm_kwargs.pop("lora_local_path", None)
    max_lora_rank = llm_kwargs.pop("max_lora_rank", None)
    if max_length is not None:
        llm_kwargs["max_model_len"] = max_length
    if lora_path is not None:
        llm_kwargs["enable_lora"] = True
    if max_lora_rank is not None:
        llm_kwargs["max_lora_rank"] = max_lora_rank
    return VLLMRunnerSpec(model_name=model_name, llm_kwargs=llm_kwargs, max_lora_rank=max_lora_rank), str(lora_path) if lora_path is not None else None


class VLLMRunner:
    def __init__(self, model_args: dict[str, Any]) -> None:
        from vllm import LLM

        self.spec, _ = build_vllm_runner_spec(model_args)
        self._llm = LLM(model=self.spec.model_name, **self.spec.llm_kwargs)
        self._request_counter = 0
        self._tokenizer: object | None = None

    def matches(self, model_args: dict[str, Any]) -> bool:
        candidate_spec, _ = build_vllm_runner_spec(model_args)
        return candidate_spec == self.spec

    def generate_raw_eval(
        self,
        *,
        model_args: dict[str, Any],
        dataset_path: str | Path,
        task_name: str,
        batch_size: int | str,
        limit: int | None,
        max_gen_toks: int,
        samples_per_problem: int = 1,
        sampling_temperature: float | None = None,
        sampling_top_p: float | None = None,
        prompt_style: str = "justrl_math",
        use_chat_template: bool = False,
        system_prompt: str | None = None,
        assistant_prefill: str | None = None,
    ) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if batch_size == "auto":
            raise ValueError("当前评测不支持 batch_size=auto，请显式传整数 batch_size。")
        if not self.matches(model_args):
            raise ValueError("复用 vLLM runner 时发现 base model 或 server 参数不一致，拒绝静默复用。")

        rows = read_jsonl(dataset_path)
        if limit is not None:
            rows = rows[:limit]
        prompts = [
            self._build_prompt(
                str(row["question"]),
                prompt_style=prompt_style,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                assistant_prefill=assistant_prefill,
            )
            for row in rows
        ]
        _, lora_path = build_vllm_runner_spec(model_args)

        if samples_per_problem > 1:
            temperature = 0.6 if sampling_temperature is None else sampling_temperature
            top_p = 0.95 if sampling_top_p is None else sampling_top_p
            if temperature <= 0.0:
                raise ValueError("samples_per_problem > 1 时 temperature 必须大于 0。")
        else:
            temperature = 0.0
            top_p = 1.0

        sampling_params = SamplingParams(
            n=samples_per_problem,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_toks,
            stop=["</s>", "<|im_end|>"],
        )
        lora_request = None
        if lora_path is not None:
            self._request_counter += 1
            adapter_name = f"eval_adapter_{Path(lora_path).name}_{self._request_counter}"
            lora_request = LoRARequest(adapter_name, self._request_counter, lora_path)

        started_at = time.perf_counter()
        outputs = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        total_seconds = time.perf_counter() - started_at

        samples: list[dict[str, Any]] = []
        for row, prompt, output in zip(rows, prompts, outputs, strict=True):
            generations = [candidate.text for candidate in output.outputs]
            samples.append(
                {
                    "id": str(row.get("id", "")),
                    "doc": row,
                    "prompt": prompt,
                    "resps": generations,
                    "metrics": ["raw_count"],
                }
            )

        return {
            "runner": "vllm_raw",
            "prompt_config": {
                "prompt_style": prompt_style,
                "use_chat_template": use_chat_template,
                "system_prompt": system_prompt,
                "assistant_prefill": assistant_prefill,
            },
            "task_name": task_name,
            "samples_per_problem": samples_per_problem,
            "samples": {task_name: samples},
            "timing": {"total_seconds": total_seconds},
        }

    def close(self) -> None:
        engine = getattr(self._llm, "llm_engine", None)
        if engine is None:
            return
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _get_tokenizer(self) -> object:
        if self._tokenizer is not None:
            return self._tokenizer
        get_tokenizer = getattr(self._llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            self._tokenizer = get_tokenizer()
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name, trust_remote_code=True)
        return self._tokenizer

    def _build_prompt(
        self,
        question: str,
        *,
        prompt_style: str,
        use_chat_template: bool,
        system_prompt: str | None,
        assistant_prefill: str | None,
    ) -> str:
        prompt = build_math_prompt(question, style=prompt_style)  # type: ignore[arg-type]
        if not use_chat_template:
            return prompt + (assistant_prefill or "")
        return render_chat_prompt(
            self._get_tokenizer(),
            prompt,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )


def run_vllm_raw_eval(
    *,
    model_args: dict[str, Any],
    dataset_path: str | Path,
    task_name: str,
    batch_size: int | str,
    limit: int | None,
    max_gen_toks: int,
    samples_per_problem: int = 1,
    sampling_temperature: float | None = None,
    sampling_top_p: float | None = None,
    prompt_style: str = "justrl_math",
    use_chat_template: bool = False,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
) -> dict[str, Any]:
    runner = VLLMRunner(model_args)
    try:
        return runner.generate_raw_eval(
            model_args=model_args,
            dataset_path=dataset_path,
            task_name=task_name,
            batch_size=batch_size,
            limit=limit,
            max_gen_toks=max_gen_toks,
            samples_per_problem=samples_per_problem,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            prompt_style=prompt_style,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )
    finally:
        runner.close()
