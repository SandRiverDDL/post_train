from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.data import make_sft_record, split_train_dev, summarize_sft_dataset


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
