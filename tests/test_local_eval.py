from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.local_eval import build_official_result, rate_stderr, run_official_eval


class LocalEvalTest(unittest.TestCase):
    def test_rate_stderr_handles_empty_input(self) -> None:
        self.assertEqual(rate_stderr(0.5, 0), 0.0)

    def test_rate_stderr_matches_binomial_standard_error(self) -> None:
        stderr = rate_stderr(0.75, 100)
        self.assertAlmostEqual(stderr, math.sqrt(0.75 * 0.25 / 100), places=8)

    def test_build_official_result_includes_summary_and_predictions(self) -> None:
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
            "Reasoning\n\nFinal answer: \\boxed{2}",
            "Reasoning only",
        ]

        result = build_official_result(
            rows,
            generations,
            model_name="outputs/demo",
            dataset_path=Path("data/eval/toy.jsonl"),
        )

        self.assertEqual(result["metrics"]["model"], "outputs/demo")
        self.assertEqual(result["metrics"]["dataset"], "data/eval/toy.jsonl")
        self.assertEqual(result["metrics"]["samples"], 2)
        self.assertEqual(result["metrics"]["format_success_rate"], 0.5)
        self.assertEqual(result["metrics"]["parse_success_rate"], 0.5)
        self.assertEqual(result["metrics"]["normalized_accuracy"], 0.5)
        self.assertIn("format_success_stderr", result["metrics"])
        self.assertIn("parse_success_stderr", result["metrics"])
        self.assertIn("normalized_accuracy_stderr", result["metrics"])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["predictions"][0]["predicted_answer"], "2")
        self.assertTrue(result["predictions"][0]["format_ok"])
        self.assertFalse(result["predictions"][1]["extract_ok"])

    def test_run_official_eval_can_disable_preview(self) -> None:
        rows = [{"id": "1", "question": "1+1=?", "final_answer": "2"}]
        fake_result = {
            "metrics": {
                "model": "outputs/demo",
                "dataset": "data/eval/toy.jsonl",
                "samples": 1,
                "format_success_rate": 1.0,
                "format_success_stderr": 0.0,
                "parse_success_rate": 1.0,
                "parse_success_stderr": 0.0,
                "normalized_accuracy": 1.0,
                "normalized_accuracy_stderr": 0.0,
            },
            "predictions": [],
        }

        with (
            patch("rl.local_eval.read_jsonl", return_value=rows),
            patch("rl.local_eval.resolve_model_args", return_value={"pretrained": "base"}),
            patch("rl.local_eval._generate_with_hf", return_value=["Final answer: \\boxed{2}"]),
            patch("rl.local_eval.build_official_result", return_value=fake_result),
            patch("rl.local_eval._preview_predictions") as preview_mock,
        ):
            result = run_official_eval(
                backend="hf",
                requested_model="outputs/demo",
                base_model="Qwen/demo",
                dataset_path=Path("data/eval/toy.jsonl"),
                limit=None,
                batch_size=1,
                max_new_tokens=32,
                max_length=128,
                device="cpu",
                attn_implementation="sdpa",
                gpu_memory_utilization=0.5,
                preview_count=0,
            )

        self.assertEqual(result, fake_result)
        preview_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
