from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.data import (
    prepare_benchmark_artifact,
    make_sft_record,
    prepare_sampled_eval_artifact,
    prepare_stratified_math500_dev_artifact,
    sample_rows,
    split_train_dev,
    summarize_sft_dataset,
)


class DataPipelineTest(unittest.TestCase):
    def test_make_sft_record_builds_solution_from_answer_only(self) -> None:
        row = {
            "problem": "Compute 1/3.",
            "answer": "\\frac{1}{3}",
        }
        record = make_sft_record(row, 0, source="toy")
        self.assertEqual(record["final_answer"], "\\frac{1}{3}")
        self.assertEqual(record["solution"], "\\boxed{\\frac{1}{3}}")

    def test_make_sft_record_prefers_explicit_answer(self) -> None:
        row = {
            "question": "Compute 2+3.",
            "solution": "A long derivation without explicit final marker.",
            "answer": "5",
        }
        record = make_sft_record(row, 1, source="toy")
        self.assertEqual(record["final_answer"], "5")
        self.assertIn("\\boxed{5}", record["solution"])

    def test_split_train_dev_is_deterministic(self) -> None:
        rows = [
            {"id": str(index), "question": f"q{index}", "solution": "\\boxed{1}", "final_answer": "1"}
            for index in range(10)
        ]
        train_rows, dev_rows = split_train_dev(rows, train_size=6, dev_size=2, seed=7)
        self.assertEqual(len(train_rows), 6)
        self.assertEqual(len(dev_rows), 2)
        self.assertEqual([row["id"] for row in train_rows], ["8", "3", "1", "4", "7", "0"])
        self.assertEqual([row["id"] for row in dev_rows], ["9", "6"])

    def test_sample_rows_is_deterministic(self) -> None:
        rows = [{"id": str(index)} for index in range(6)]
        sampled = sample_rows(rows, sample_size=3, seed=7)
        self.assertEqual([row["id"] for row in sampled], ["4", "0", "5"])

    def test_prepare_sampled_eval_artifact_builds_eval_rows(self) -> None:
        raw_rows = [
            {"problem": "1+1=?", "answer": "2"},
            {"problem": "2+2=?", "answer": "4"},
            {"problem": "3+3=?", "answer": "6"},
        ]
        with patch("post_train.data.load_dataset_rows", return_value=raw_rows):
            rows = prepare_sampled_eval_artifact(
                dataset_name="toy",
                split="test",
                source="toy",
                sample_size=2,
                seed=5,
            )
        self.assertEqual(len(rows), 2)
        self.assertIn("question", rows[0])
        self.assertIn("final_answer", rows[0])
        self.assertEqual(rows[0]["meta"]["source"], "toy")

    def test_prepare_stratified_math500_dev_artifact_preserves_level_ratio(self) -> None:
        raw_rows = [
            {"problem": "q1", "answer": "1", "level": 1},
            {"problem": "q2", "answer": "2", "level": 1},
            {"problem": "q3", "answer": "3", "level": 2},
            {"problem": "q4", "answer": "4", "level": 2},
            {"problem": "q5", "answer": "5", "level": 2},
            {"problem": "q6", "answer": "6", "level": 3},
        ]
        with patch("post_train.data.load_dataset_rows", return_value=raw_rows):
            rows, report = prepare_stratified_math500_dev_artifact(
                sample_size=4,
                seed=5,
            )
        self.assertEqual(len(rows), 4)
        self.assertEqual(report["level_distribution"], {1: 2, 2: 3, 3: 1})
        self.assertEqual(report["selected_level_distribution"], {1: 1, 2: 2, 3: 1})
        self.assertEqual({row["meta"]["source"] for row in rows}, {"math500"})
        self.assertTrue(all("level" in row["meta"] for row in rows))

    def test_prepare_benchmark_artifact_supports_aime25_fields(self) -> None:
        raw_rows = [
            {"id": "a-1", "problem": "Compute 1+1.", "answer": "2"},
        ]
        with patch("post_train.data.load_dataset_rows", return_value=raw_rows):
            rows = prepare_benchmark_artifact(
                dataset_name="math-ai/aime25",
                split="test",
                source="aime25",
            )
        self.assertEqual(rows[0]["id"], "a-1")
        self.assertEqual(rows[0]["question"], "Compute 1+1.")
        self.assertEqual(rows[0]["final_answer"], "2")
        self.assertEqual(rows[0]["meta"]["source"], "aime25")

    def test_summarize_sft_dataset_counts_metrics(self) -> None:
        rows = [
            {
                "id": "1",
                "question": "1+1=?",
                "solution": "Reasoning\n\n\\boxed{2}",
                "final_answer": "2",
            },
            {
                "id": "2",
                "question": "2+2=?",
                "solution": "Reasoning only",
                "final_answer": "4",
            },
            {
                "id": "3",
                "question": "3+3=?",
                "solution": "Reasoning\n\n\\boxed{6}",
                "final_answer": "",
            },
        ]

        summary = summarize_sft_dataset(rows)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["boxed"], 2)
        self.assertAlmostEqual(summary["boxed_rate"], 2 / 3)
        self.assertEqual(summary["parse_success"], 2)
        self.assertAlmostEqual(summary["parse_success_rate"], 2 / 3)
        self.assertEqual(summary["consistent_final_answer"], 1)
        self.assertEqual(summary["empty_final_answer"], 1)


if __name__ == "__main__":
    unittest.main()
