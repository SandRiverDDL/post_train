from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_eval_config
from post_train.eval import (
    build_eval_result,
    normalize_batch_settings,
    process_results_stub,
    rate_stderr,
    resolve_eval_tasks,
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
        self.assertEqual(result["metrics"]["boxed_rate"], 0.5)
        self.assertEqual(result["metrics"]["parse_success_rate"], 0.5)
        self.assertEqual(result["metrics"]["normalized_accuracy"], 0.5)
        self.assertIn("avg_output_tokens", result["metrics"])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["predictions"][0]["predicted_answer"], "2")
        self.assertTrue(result["predictions"][0]["boxed"])
        self.assertFalse(result["predictions"][1]["parse_ok"])

    def test_normalize_batch_settings_ignores_max_batch_when_not_auto(self) -> None:
        batch_size, max_batch_size = normalize_batch_settings(6, 12)
        self.assertEqual(batch_size, 6)
        self.assertIsNone(max_batch_size)

    def test_normalize_batch_settings_preserves_max_batch_for_auto(self) -> None:
        batch_size, max_batch_size = normalize_batch_settings("auto", 12)
        self.assertEqual(batch_size, "auto")
        self.assertEqual(max_batch_size, 12)

    def test_process_results_stub_does_not_parse_answers(self) -> None:
        result = process_results_stub({"question": "1+1"}, ["\\boxed{2}"])
        self.assertEqual(result, {"raw_count": 1.0})

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
                        "max_batch_size: 8",
                        "max_new_tokens: 256",
                        "max_seq_length: 1024",
                        "device: cuda",
                        "attn_implementation: sdpa",
                        "gpu_memory_utilization: 0.7",
                        "seed: 42",
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

        self.assertEqual(len(cfg.tasks), 2)
        self.assertEqual(cfg.tasks[0].name, "gsm8k")
        self.assertEqual(cfg.tasks[1].name, "math500")

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


if __name__ == "__main__":
    unittest.main()
