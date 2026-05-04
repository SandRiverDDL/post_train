from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_eval_config
from post_train.eval import (
    build_eval_result,
    EvalRunner,
    EvalRunOverrides,
    rate_stderr,
    resolve_eval_tasks,
    resolve_task_output_paths,
    result_from_vllm_raw_logs,
    run_eval_task,
    run_vllm_raw_eval,
    VLLMRunner,
    summarize_metrics_for_console,
    write_raw_eval_result,
)
from post_train.eval.vllm import build_vllm_runner_spec, describe_model_resolution, resolve_model_args


class EvalPipelineTest(unittest.TestCase):
    def test_rate_stderr_handles_empty_input(self) -> None:
        self.assertEqual(rate_stderr(0.5, 0), 0.0)

    def test_rate_stderr_matches_binomial_standard_error(self) -> None:
        stderr = rate_stderr(0.75, 100)
        self.assertAlmostEqual(stderr, math.sqrt(0.75 * 0.25 / 100), places=8)

    def test_build_eval_result_includes_summary_and_predictions(self) -> None:
        rows = [
            {
                "id": "1",
                "question": "1+1=?",
                "final_answer": "2",
            },
            {
                "id": "2",
                "question": "2+2=?",
                "final_answer": "4",
            },
        ]
        generations = [
            "Reasoning\n\n\\boxed{2}",
            "Reasoning only",
        ]

        result = build_eval_result(
            rows,
            generations,
            model_name="outputs/demo",
            dataset_name="data/eval/toy.jsonl",
        )

        self.assertEqual(result["metrics"]["model"], "outputs/demo")
        self.assertEqual(result["metrics"]["dataset"], "data/eval/toy.jsonl")
        self.assertEqual(result["metrics"]["samples"], 2)
        self.assertEqual(result["metrics"]["total_generations"], 2)
        self.assertEqual(result["metrics"]["boxed_rate"], 0.5)
        self.assertEqual(result["metrics"]["parse_success_rate"], 0.5)
        self.assertEqual(result["metrics"]["normalized_accuracy"], 0.5)
        self.assertEqual(result["metrics"]["pass_at_1"], 0.5)
        self.assertIn("avg_output_tokens", result["metrics"])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["predictions"][0]["predicted_answer"], "2")
        self.assertTrue(result["predictions"][0]["boxed"])
        self.assertEqual(result["predictions"][0]["sample_index"], 1)
        self.assertFalse(result["predictions"][1]["parse_ok"])

    def test_build_eval_result_supports_multi_sample_pass_at_1(self) -> None:
        rows = [
            {"id": "1", "question": "q1", "final_answer": "2"},
            {"id": "2", "question": "q2", "final_answer": "4"},
        ]
        generations = [
            ["\\boxed{2}", "wrong", "\\boxed{2}", "wrong"],
            ["wrong", "wrong", "wrong", "\\boxed{4}"],
        ]

        result = build_eval_result(
            rows,
            generations,
            model_name="outputs/demo",
            dataset_name="data/eval/aime25_test.jsonl",
        )

        self.assertEqual(result["metrics"]["samples"], 2)
        self.assertEqual(result["metrics"]["total_generations"], 8)
        self.assertEqual(result["metrics"]["samples_per_problem"], 4.0)
        self.assertAlmostEqual(result["metrics"]["pass_at_1"], 0.375)
        self.assertAlmostEqual(result["metrics"]["normalized_accuracy"], 0.375)
        self.assertEqual(len(result["predictions"]), 8)
        self.assertEqual(result["predictions"][4]["id"], "2")
        self.assertEqual(result["predictions"][4]["sample_index"], 1)

    def test_summarize_metrics_for_console_prefers_pass_at_1(self) -> None:
        summary = summarize_metrics_for_console(
            {
                "pass_at_1": 0.1,
                "pass_at_1_stderr": 0.02,
                "normalized_accuracy": 0.1,
                "normalized_accuracy_stderr": 0.02,
                "boxed_rate": 0.95,
                "parse_success_rate": 0.94,
                "avg_output_tokens": 400.0,
                "samples": 30,
            }
        )
        self.assertEqual(
            summary,
            {
                "model": None,
                "dataset": None,
                "samples": 30,
                "pass_at_1": 0.1,
                "pass_at_1_stderr": 0.02,
                "boxed_rate": 0.95,
                "parse_success_rate": 0.94,
                "avg_output_tokens": 400.0,
            },
        )

    def test_summarize_metrics_for_console_falls_back_to_accuracy(self) -> None:
        summary = summarize_metrics_for_console(
            {
                "normalized_accuracy": 0.6,
                "normalized_accuracy_stderr": 0.03,
                "boxed_rate": 0.98,
                "parse_success_rate": 0.97,
                "avg_output_tokens": 120.0,
                "samples": 150,
            }
        )
        self.assertEqual(
            summary,
            {
                "model": None,
                "dataset": None,
                "samples": 150,
                "normalized_accuracy": 0.6,
                "normalized_accuracy_stderr": 0.03,
                "boxed_rate": 0.98,
                "parse_success_rate": 0.97,
                "avg_output_tokens": 120.0,
            },
        )

    def test_load_eval_config_supports_task_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: Qwen/Qwen2.5-Math-1.5B",
                        "output_dir: outputs/eval",
                        "backend: vllm",
                        "batch_size: auto",
                        "max_new_tokens: 256",
                        "max_seq_length: 1024",
                        "device: cuda",
                        "attn_implementation: sdpa",
                        "gpu_memory_utilization: 0.7",
                        "max_lora_rank: 32",
                        "seed: 42",
                        "tasks:",
                        "  - name: gsm8k",
                        "    dataset_path: data/eval/gsm8k_test.jsonl",
                        "  - name: math500",
                        "    dataset_path: data/eval/math500_test.jsonl",
                        "  - name: aime25",
                        "    dataset_path: data/eval/aime25_test.jsonl",
                        "    samples_per_problem: 4",
                        "    sampling_temperature: 0.6",
                        "    sampling_top_p: 0.95",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_eval_config(config_path)

        self.assertEqual(len(cfg.tasks), 3)
        self.assertEqual(cfg.backend, "vllm")
        self.assertEqual(cfg.tasks[0].name, "gsm8k")
        self.assertEqual(cfg.tasks[2].name, "aime25")
        self.assertEqual(cfg.tasks[2].samples_per_problem, 4)
        self.assertEqual(cfg.tasks[2].sampling_temperature, 0.6)
        self.assertEqual(cfg.tasks[2].sampling_top_p, 0.95)
        self.assertEqual(cfg.max_lora_rank, 32)

    def test_resolve_eval_tasks_filters_requested_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: Qwen/Qwen2.5-Math-1.5B",
                        "output_dir: outputs/eval",
                        "tasks:",
                        "  - name: gsm8k",
                        "    dataset_path: data/eval/gsm8k_test.jsonl",
                        "  - name: math500",
                        "    dataset_path: data/eval/math500_test.jsonl",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_eval_config(config_path)

        tasks = resolve_eval_tasks(cfg, requested_tasks=["math500"])
        self.assertEqual([task.name for task in tasks], ["math500"])

    def test_load_eval_aime_config_supports_both_tasks(self) -> None:
        cfg = load_eval_config("configs/eval/aime.yaml")

        self.assertEqual([task.name for task in cfg.tasks], ["aime24", "aime25"])
        self.assertGreater(cfg.tasks[0].samples_per_problem, 0)
        self.assertEqual(cfg.tasks[0].samples_per_problem, cfg.tasks[1].samples_per_problem)
        self.assertEqual(cfg.tasks[1].sampling_temperature, 0.6)

    def test_resolve_eval_tasks_filters_aime_tasks_from_combined_config(self) -> None:
        cfg = load_eval_config("configs/eval/aime.yaml")

        tasks = resolve_eval_tasks(cfg, requested_tasks=["aime24"])

        self.assertEqual([task.name for task in tasks], ["aime24"])

    def test_write_raw_eval_result_serializes_callable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "raw.json"
            raw_result = {
                "configs": {
                    "gsm8k": {
                        "doc_to_text": lambda doc: doc["question"],
                    }
                },
                "samples": [
                    {
                        "value": "ok",
                    }
                ],
            }

            write_raw_eval_result(output_path, raw_result)

            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIsInstance(loaded["configs"]["gsm8k"]["doc_to_text"], str)
        self.assertEqual(loaded["samples"][0]["value"], "ok")

    def test_resolve_model_args_passes_max_lora_rank_for_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_dir = Path(tmp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")

            model_args = resolve_model_args(
                str(adapter_dir),
                "Qwen/Qwen2.5-Math-1.5B",
                backend="vllm",
                max_length=1536,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.7,
                max_lora_rank=32,
            )

        self.assertEqual(model_args["pretrained"], "Qwen/Qwen2.5-Math-1.5B")
        self.assertEqual(model_args["lora_local_path"], str(adapter_dir))
        self.assertEqual(model_args["max_lora_rank"], 32)

    def test_describe_model_resolution_marks_adapter_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_dir = Path(tmp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            model_args = resolve_model_args(
                str(adapter_dir),
                "/tmp/base",
                backend="vllm",
                max_length=1536,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.7,
                max_lora_rank=64,
            )

        resolution = describe_model_resolution(str(adapter_dir), "/tmp/base", model_args)

        self.assertTrue(resolution["enable_lora"])
        self.assertTrue(resolution["target_is_adapter_dir"])
        self.assertEqual(resolution["pretrained"], "/tmp/base")
        self.assertEqual(resolution["lora_local_path"], str(adapter_dir))
        self.assertEqual(resolution["max_lora_rank"], 64)

    def test_run_eval_task_returns_model_resolution_for_adapter_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "toy.jsonl"
            dataset_path.write_text(
                json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            adapter_dir = tmp_path / "checkpoint-10"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            config_path = tmp_path / "eval.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/base",
                        f"dataset_path: {dataset_path}",
                        f"output_dir: {tmp_path / 'outputs'}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_eval_config(config_path)

            with patch("post_train.eval.runner.run_vllm_raw_eval", return_value={"samples": {"toy": []}, "timing": {"total_seconds": 1.0}}), patch(
                "post_train.eval.runner.result_from_vllm_raw_logs",
                return_value={"metrics": {"normalized_accuracy": 0.0}},
            ):
                run_result = run_eval_task(
                    model_name=str(adapter_dir),
                    task=cfg.tasks[0],
                    eval_cfg=cfg,
                    batch_size=1,
                    max_new_tokens=64,
                    limit=1,
                    output_dir=tmp_path / "outputs",
                    max_lora_rank=32,
                )

        self.assertTrue(run_result["model_resolution"]["enable_lora"])
        self.assertEqual(run_result["model_resolution"]["pretrained"], "/tmp/base")
        self.assertEqual(run_result["model_resolution"]["lora_local_path"], str(adapter_dir))

    def test_eval_runner_run_task_matches_run_eval_task_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "toy.jsonl"
            dataset_path.write_text(
                json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "eval.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/base",
                        f"dataset_path: {dataset_path}",
                        f"output_dir: {tmp_path / 'outputs'}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_eval_config(config_path)

            with patch("post_train.eval.runner.run_vllm_raw_eval", return_value={"samples": {"toy": []}, "timing": {"total_seconds": 1.0}}), patch(
                "post_train.eval.runner.result_from_vllm_raw_logs",
                return_value={"metrics": {"normalized_accuracy": 0.0}},
            ):
                run_result = EvalRunner(cfg, model_name="/tmp/base").run_task(
                    cfg.tasks[0],
                    overrides=EvalRunOverrides(
                        batch_size=2,
                        max_new_tokens=64,
                        limit=1,
                        output_dir=tmp_path / "custom_outputs",
                        max_lora_rank=32,
                    ),
                )

        self.assertEqual(set(run_result), {"task_name", "raw_result", "result", "result_path", "raw_result_path", "model_resolution"})
        self.assertEqual(run_result["task_name"], "toy")
        self.assertIn("custom_outputs", run_result["result_path"])

    def test_result_from_vllm_raw_logs_matches_current_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "toy.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False),
                        json.dumps({"id": "2", "question": "2+2=?", "final_answer": "4"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            raw_result = {
                "runner": "vllm_raw",
                "samples": {
                    "toy": [
                        {"resps": ["\\boxed{2}"]},
                        {"resps": ["not parsed"]},
                    ]
                },
                "timing": {"total_seconds": 2.0},
            }

            result = result_from_vllm_raw_logs(
                raw_result,
                task_name="toy",
                dataset_path=dataset_path,
                model_name="outputs/demo",
            )

        self.assertEqual(result["metrics"]["runner"], "vllm_raw")
        self.assertEqual(result["metrics"]["samples"], 2)
        self.assertEqual(result["metrics"]["total_generations"], 2)
        self.assertEqual(result["metrics"]["boxed_rate"], 0.5)
        self.assertEqual(result["metrics"]["samples_per_second"], 1.0)
        self.assertEqual(result["metrics"]["generations_per_second"], 1.0)
        self.assertEqual(len(result["predictions"]), 2)

    def test_resolve_task_output_paths_uses_model_scoped_directory(self) -> None:
        task = type("Task", (), {"name": "gsm8k", "output_path": None, "raw_output_path": None})()
        final_output, raw_output = resolve_task_output_paths(
            task,
            output_dir=Path("outputs/eval"),
            model_name="outputs/stage1_sft/checkpoint-50",
        )
        self.assertEqual(final_output, Path("outputs/eval/stage1_sft/checkpoint-50/gsm8k/result.json"))
        self.assertEqual(raw_output, Path("outputs/eval/stage1_sft/checkpoint-50/gsm8k/raw.json"))

    def test_resolve_task_output_paths_rejects_file_override(self) -> None:
        task = type(
            "Task",
            (),
            {"name": "gsm8k", "output_path": Path("outputs/eval/custom.json"), "raw_output_path": None},
        )()

        with self.assertRaisesRegex(ValueError, "--output 必须是目录路径"):
            resolve_task_output_paths(
                task,
                output_dir=Path("outputs/eval"),
                model_name="outputs/stage1_sft/checkpoint-50",
            )

    def test_resolve_task_output_paths_external_model_uses_external_prefix(self) -> None:
        task = type("Task", (), {"name": "math500", "output_path": None, "raw_output_path": None})()
        final_output, raw_output = resolve_task_output_paths(
            task,
            output_dir=Path("outputs/eval"),
            model_name="Qwen/Qwen2.5-Math-1.5B",
        )
        self.assertEqual(
            final_output,
            Path("outputs/eval/external/Qwen/Qwen2.5-Math-1.5B/math500/result.json"),
        )
        self.assertEqual(
            raw_output,
            Path("outputs/eval/external/Qwen/Qwen2.5-Math-1.5B/math500/raw.json"),
        )

    def test_run_vllm_raw_eval_calls_generate_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "toy.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False),
                        json.dumps({"id": "2", "question": "2+2=?", "final_answer": "4"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calls: list[dict[str, object]] = []

            class FakeEngine:
                def shutdown(self) -> None:
                    calls.append({"shutdown": True})

            class FakeLLM:
                def __init__(self, model=None, **kwargs) -> None:
                    self.model = model
                    self.kwargs = kwargs
                    self.llm_engine = FakeEngine()
                    calls.append({"init_model": model, "init_kwargs": kwargs})

                def generate(self, prompts, sampling_params=None, use_tqdm=False, lora_request=None):
                    calls.append(
                        {
                            "prompt_count": len(prompts),
                            "use_tqdm": use_tqdm,
                            "lora_request": lora_request,
                        }
                    )
                    return [
                        SimpleNamespace(outputs=[SimpleNamespace(text="\\boxed{2}")]),
                        SimpleNamespace(outputs=[SimpleNamespace(text="\\boxed{4}")]),
                    ]

                def get_tokenizer(self):
                    class FakeTokenizer:
                        chat_template = "dummy"

                        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
                            rendered = ""
                            for message in messages:
                                rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                            if add_generation_prompt:
                                rendered += "<|im_start|>assistant\n"
                            return rendered

                    return FakeTokenizer()

            class FakeSamplingParams:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    calls.append({"sampling_params": kwargs})

            fake_vllm = ModuleType("vllm")
            fake_vllm.LLM = FakeLLM
            fake_vllm.SamplingParams = FakeSamplingParams
            fake_lora_request_module = ModuleType("vllm.lora.request")
            fake_lora_request_module.LoRARequest = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

            with patch.dict(
                sys.modules,
                {
                    "vllm": fake_vllm,
                    "vllm.lora.request": fake_lora_request_module,
                },
            ):
                result = run_vllm_raw_eval(
                    model_args={
                        "pretrained": "Qwen/Qwen2.5-Math-1.5B",
                        "dtype": "auto",
                        "trust_remote_code": True,
                        "max_length": 1536,
                        "gpu_memory_utilization": 0.7,
                    },
                    dataset_path=dataset_path,
                    task_name="toy",
                    batch_size=1,
                    limit=None,
                    max_gen_toks=128,
                    samples_per_problem=4,
                    sampling_temperature=0.6,
                    sampling_top_p=0.95,
                )

        self.assertEqual(calls[2]["prompt_count"], 2)
        self.assertEqual(calls[1]["sampling_params"]["n"], 4)
        self.assertEqual(calls[1]["sampling_params"]["temperature"], 0.6)
        self.assertEqual(calls[1]["sampling_params"]["top_p"], 0.95)
        self.assertEqual(result["runner"], "vllm_raw")
        self.assertEqual(len(result["samples"]["toy"]), 2)
        self.assertTrue(calls[-1]["shutdown"])

    def test_run_vllm_raw_eval_can_apply_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "toy.jsonl"
            dataset_path.write_text(
                json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            calls: list[dict[str, object]] = []

            class FakeEngine:
                def shutdown(self) -> None:
                    calls.append({"shutdown": True})

            class FakeTokenizer:
                chat_template = "dummy"

                def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
                    rendered = ""
                    for message in messages:
                        rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                    if add_generation_prompt:
                        rendered += "<|im_start|>assistant\n"
                    return rendered

            class FakeLLM:
                def __init__(self, model=None, **kwargs) -> None:
                    self.llm_engine = FakeEngine()

                def get_tokenizer(self):
                    return FakeTokenizer()

                def generate(self, prompts, sampling_params=None, use_tqdm=False, lora_request=None):
                    calls.append({"prompts": prompts})
                    return [SimpleNamespace(outputs=[SimpleNamespace(text="\\boxed{2}")])]

            class FakeSamplingParams:
                def __init__(self, **kwargs) -> None:
                    pass

            fake_vllm = ModuleType("vllm")
            fake_vllm.LLM = FakeLLM
            fake_vllm.SamplingParams = FakeSamplingParams
            fake_lora_request_module = ModuleType("vllm.lora.request")
            fake_lora_request_module.LoRARequest = lambda *args, **kwargs: SimpleNamespace()

            with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.lora.request": fake_lora_request_module}):
                result = run_vllm_raw_eval(
                    model_args={"pretrained": "Qwen/Qwen3-1.7B", "max_length": 1536},
                    dataset_path=dataset_path,
                    task_name="toy",
                    batch_size=1,
                    limit=None,
                    max_gen_toks=128,
                    use_chat_template=True,
                    system_prompt="",
                )

        prompt = calls[0]["prompts"][0]
        self.assertIn("<|im_start|>system\n<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\n1+1=?", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertTrue(result["prompt_config"]["use_chat_template"])

    def test_vllm_runner_reuses_llm_and_switches_lora_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "toy.jsonl"
            adapter_a = Path(tmp_dir) / "checkpoint-10"
            adapter_b = Path(tmp_dir) / "checkpoint-20"
            for adapter_dir in (adapter_a, adapter_b):
                adapter_dir.mkdir()
                (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
                (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
            dataset_path.write_text(json.dumps({"id": "1", "question": "1+1=?", "final_answer": "2"}, ensure_ascii=False) + "\n", encoding="utf-8")

            calls: list[dict[str, object]] = []

            class FakeEngine:
                def shutdown(self) -> None:
                    calls.append({"shutdown": True})

            class FakeLLM:
                def __init__(self, model=None, **kwargs) -> None:
                    self.llm_engine = FakeEngine()
                    calls.append({"init_model": model, "init_kwargs": kwargs})

                def generate(self, prompts, sampling_params=None, use_tqdm=False, lora_request=None):
                    calls.append({"lora_request": lora_request, "prompt_count": len(prompts)})
                    return [SimpleNamespace(outputs=[SimpleNamespace(text="\\boxed{2}")])]

            class FakeSamplingParams:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

            class FakeLoRARequest:
                def __init__(self, adapter_name, adapter_id, lora_path) -> None:
                    self.adapter_name = adapter_name
                    self.adapter_id = adapter_id
                    self.lora_path = lora_path

            fake_vllm = ModuleType("vllm")
            fake_vllm.LLM = FakeLLM
            fake_vllm.SamplingParams = FakeSamplingParams
            fake_lora_request_module = ModuleType("vllm.lora.request")
            fake_lora_request_module.LoRARequest = FakeLoRARequest

            base_model_args = resolve_model_args(
                str(adapter_a),
                "Qwen/Qwen2.5-Math-1.5B",
                backend="vllm",
                max_length=1536,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.7,
                max_lora_rank=64,
            )
            alt_model_args = resolve_model_args(
                str(adapter_b),
                "Qwen/Qwen2.5-Math-1.5B",
                backend="vllm",
                max_length=1536,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.7,
                max_lora_rank=64,
            )

            with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.lora.request": fake_lora_request_module}):
                runner = VLLMRunner(base_model_args)
                try:
                    runner.generate_raw_eval(
                        model_args=base_model_args,
                        dataset_path=dataset_path,
                        task_name="toy",
                        batch_size=1,
                        limit=None,
                        max_gen_toks=64,
                    )
                    runner.generate_raw_eval(
                        model_args=alt_model_args,
                        dataset_path=dataset_path,
                        task_name="toy",
                        batch_size=1,
                        limit=None,
                        max_gen_toks=64,
                    )
                finally:
                    runner.close()

        init_calls = [item for item in calls if "init_model" in item]
        generate_calls = [item for item in calls if "lora_request" in item]
        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(generate_calls), 2)
        self.assertNotEqual(generate_calls[0]["lora_request"].adapter_name, generate_calls[1]["lora_request"].adapter_name)
        self.assertNotEqual(generate_calls[0]["lora_request"].adapter_id, generate_calls[1]["lora_request"].adapter_id)
        self.assertEqual(generate_calls[0]["lora_request"].lora_path, str(adapter_a))
        self.assertEqual(generate_calls[1]["lora_request"].lora_path, str(adapter_b))
        self.assertTrue(calls[-1]["shutdown"])

if __name__ == "__main__":
    unittest.main()
