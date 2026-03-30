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
    rate_stderr,
    resolve_model_args,
    resolve_eval_tasks,
    resolve_task_output_paths,
    result_from_vllm_raw_logs,
    run_vllm_raw_eval,
    summarize_metrics_for_console,
    write_raw_eval_result,
)


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

            class FakeLLM:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

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

        self.assertEqual(calls[1]["prompt_count"], 2)
        self.assertEqual(calls[0]["sampling_params"]["n"], 4)
        self.assertEqual(calls[0]["sampling_params"]["temperature"], 0.6)
        self.assertEqual(calls[0]["sampling_params"]["top_p"], 0.95)
        self.assertEqual(result["runner"], "vllm_raw")
        self.assertEqual(len(result["samples"]["toy"]), 2)

if __name__ == "__main__":
    unittest.main()
