from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys

import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lm_eval.evaluator_utils import consolidate_results, get_task_list, prepare_print_tasks

from rl.harness_tasks import (
    _extract_preview_gold_answer,
    _extract_preview_raw_generation,
    build_local_task,
    build_task_config,
    resolve_benchmark_task,
    resolve_model_args,
    strict_process_results,
)


class HarnessTaskTest(unittest.TestCase):
    def test_strict_metrics_require_boxed(self) -> None:
        doc = {"final_answer": "\\frac{14}{3}"}
        metrics = strict_process_results(doc, ["The answer is: \\frac{14}{3}"])
        self.assertEqual(metrics["format_success"], 0.0)
        self.assertEqual(metrics["parse_success"], 0.0)
        self.assertEqual(metrics["normalized_accuracy"], 0.0)

    def test_adapter_model_args(self) -> None:
        adapter_dir = ROOT / "tmp_adapter"
        adapter_dir.mkdir(exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        try:
            model_args = resolve_model_args(
                str(adapter_dir),
                "base-model",
                backend="hf",
                max_length=1024,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.85,
            )
            self.assertEqual(model_args["pretrained"], "base-model")
            self.assertEqual(model_args["peft"], str(adapter_dir))
            self.assertEqual(model_args["attn_implementation"], "sdpa")
        finally:
            (adapter_dir / "adapter_config.json").unlink(missing_ok=True)
            adapter_dir.rmdir()

    def test_vllm_adapter_model_args(self) -> None:
        adapter_dir = ROOT / "tmp_adapter"
        adapter_dir.mkdir(exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        try:
            model_args = resolve_model_args(
                str(adapter_dir),
                "base-model",
                backend="vllm",
                max_length=1024,
                device="cuda",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.85,
            )
            self.assertEqual(model_args["pretrained"], "base-model")
            self.assertEqual(model_args["lora_local_path"], str(adapter_dir))
            self.assertNotIn("device", model_args)
        finally:
            (adapter_dir / "adapter_config.json").unlink(missing_ok=True)
            adapter_dir.rmdir()

    def test_task_config_uses_local_json(self) -> None:
        config = build_task_config("data/eval/gsm8k_dev200.jsonl", "gsm8k_dev200", strict=True)
        self.assertEqual(config["dataset_path"], "json")
        self.assertEqual(config["dataset_kwargs"]["data_files"]["train"], "data/eval/gsm8k_dev200.jsonl")

    def test_resolve_native_gsm8k_task(self) -> None:
        task_name, sample_ids = resolve_benchmark_task("data/eval/gsm8k_dev200.jsonl")
        self.assertEqual(task_name, "gsm8k_cot_zeroshot")
        self.assertIsInstance(sample_ids, list)

    def test_preview_uses_native_raw_response(self) -> None:
        sample = {
            "resps": [[" 原始推理文本\nThe answer is 366.\n"]],
            "filtered_resps": ["366"],
        }
        self.assertIn("原始推理文本", _extract_preview_raw_generation(sample))

    def test_preview_uses_native_target_as_gold(self) -> None:
        sample = {"target": "推理过程\n#### 366"}
        doc = {"question": "q", "answer": "推理过程\n#### 366"}
        self.assertEqual(_extract_preview_gold_answer(sample, doc), "推理过程\n#### 366")

    def test_local_task_keeps_task_name_for_prepare_print(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "toy_eval.jsonl"
            dataset_path.write_text(
                '{"question":"1+1=?","final_answer":"Final answer: \\\\boxed{2}"}\n',
                encoding="utf-8",
            )
            cache_dir = Path(tmpdir) / "hf_cache"
            old_cache_env = os.environ.get("HF_DATASETS_CACHE")
            old_cache_config = datasets.config.HF_DATASETS_CACHE
            os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
            datasets.config.HF_DATASETS_CACHE = str(cache_dir)
            try:
                task = build_local_task(dataset_path, "toy_eval", strict=True)
            finally:
                datasets.config.HF_DATASETS_CACHE = old_cache_config
                if old_cache_env is None:
                    os.environ.pop("HF_DATASETS_CACHE", None)
                else:
                    os.environ["HF_DATASETS_CACHE"] = old_cache_env

        task_dict = {"toy_eval": task}
        eval_tasks = get_task_list(task_dict)
        self.assertEqual(eval_tasks[0].task_name, "toy_eval")
        eval_tasks[0].sample_metrics[("normalized_accuracy", "none")] = [1.0]
        eval_tasks[0].agg_metrics["normalized_accuracy,none"] = 1.0
        eval_tasks[0].sample_len = 1

        results, *_ = consolidate_results(eval_tasks)
        task_agg, _ = prepare_print_tasks(task_dict, results)
        self.assertEqual(task_agg["toy_eval"]["alias"], "toy_eval")


if __name__ == "__main__":
    unittest.main()
