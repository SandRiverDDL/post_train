from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grpo_checkpoints import (
    collect_model_dirs,
    default_json_output_path,
    default_output_path,
    main,
    parse_args,
)


class EvalGRPOCheckpointsTest(unittest.TestCase):
    def test_collect_model_dirs_sorts_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoint-50").mkdir()
            (run_dir / "checkpoint-10").mkdir()
            (run_dir / "checkpoint-100").mkdir()

            model_dirs = collect_model_dirs(run_dir, include_final=False)
            self.assertEqual([path.name for path in model_dirs], ["checkpoint-10", "checkpoint-50", "checkpoint-100"])

    def test_collect_model_dirs_can_include_final_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoint-25").mkdir()
            (run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            model_dirs = collect_model_dirs(run_dir, include_final=True)
            self.assertEqual([path.name for path in model_dirs], ["checkpoint-25", run_dir.name])

    def test_parse_args_defaults_to_eval_config(self) -> None:
        with patch.object(sys, "argv", ["eval_grpo_checkpoints.py", "--run-dir", "outputs/grpo"]):
            args = parse_args()
        self.assertEqual(args.config, "configs/eval.yaml")

    def test_default_output_path_uses_run_dir_and_dataset_stem(self) -> None:
        run_dir = Path("outputs/grpo-demo")
        dataset_path = Path("data/eval/gsm8k_dev200.jsonl")
        self.assertEqual(
            default_output_path(run_dir, dataset_path),
            run_dir / "checkpoint_eval_gsm8k_dev200.jsonl",
        )

    def test_default_json_output_path_uses_run_dir_and_dataset_stem(self) -> None:
        run_dir = Path("outputs/grpo-demo")
        dataset_path = Path("data/eval/gsm8k_dev200.jsonl")
        self.assertEqual(
            default_json_output_path(run_dir, dataset_path),
            run_dir / "checkpoint_eval_gsm8k_dev200.json",
        )

    def test_main_writes_official_metrics_jsonl_and_disables_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoint-20").mkdir()
            (run_dir / "checkpoint-5").mkdir()
            dataset_path = Path("data/eval/gsm8k_dev200.jsonl")
            fake_cfg = type(
                "Cfg",
                (),
                {
                    "eval_backend": "vllm",
                    "eval_dataset": dataset_path,
                    "eval_attn_implementation": "sdpa",
                    "eval_max_new_tokens": 256,
                    "max_seq_length": 1024,
                    "eval_gpu_memory_utilization": 0.85,
                    "model_name": "Qwen/demo",
                    "prompt_version": "v1",
                },
            )()
            official_results = [
                {
                    "metrics": {
                        "model": str(run_dir / "checkpoint-5"),
                        "dataset": str(dataset_path),
                        "samples": 200,
                        "format_success_rate": 0.8,
                        "format_success_stderr": 0.01,
                        "parse_success_rate": 0.8,
                        "parse_success_stderr": 0.01,
                        "normalized_accuracy": 0.5,
                        "normalized_accuracy_stderr": 0.02,
                    }
                },
                {
                    "metrics": {
                        "model": str(run_dir / "checkpoint-20"),
                        "dataset": str(dataset_path),
                        "samples": 200,
                        "format_success_rate": 0.9,
                        "format_success_stderr": 0.01,
                        "parse_success_rate": 0.9,
                        "parse_success_stderr": 0.01,
                        "normalized_accuracy": 0.6,
                        "normalized_accuracy_stderr": 0.02,
                    }
                },
            ]

            with (
                patch("eval_grpo_checkpoints.load_eval_config", return_value=fake_cfg),
                patch("eval_grpo_checkpoints.run_official_eval", side_effect=official_results) as eval_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "eval_grpo_checkpoints.py",
                        "--run-dir",
                        str(run_dir),
                        "--dataset",
                        str(dataset_path),
                        "--backend",
                        "hf",
                        "--device",
                        "cpu",
                    ],
                ),
            ):
                main()

            output_path = run_dir / "checkpoint_eval_gsm8k_dev200.jsonl"
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["checkpoint"] for row in rows], ["checkpoint-5", "checkpoint-20"])
            self.assertEqual(rows[0]["step"], 5)
            self.assertEqual(rows[1]["normalized_accuracy"], 0.6)
            self.assertEqual(rows[0]["dataset"], str(dataset_path))
            self.assertEqual(rows[0]["samples"], 200)
            self.assertEqual(eval_mock.call_args_list[0].kwargs["preview_count"], 0)
            self.assertEqual(eval_mock.call_args_list[0].kwargs["backend"], "hf")

            json_path = run_dir / "checkpoint_eval_gsm8k_dev200.json"
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["run_dir"], str(run_dir))
            self.assertEqual(summary["dataset"], str(dataset_path))
            self.assertEqual(len(summary["rows"]), 2)
            self.assertEqual(summary["best_checkpoint_by_normalized_accuracy"]["checkpoint"], "checkpoint-20")
            self.assertEqual(summary["best_checkpoint_by_normalized_accuracy"]["normalized_accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
