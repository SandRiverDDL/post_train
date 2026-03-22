from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.grpo_data import build_train_record, infer_source, select_rows_by_budget
from prepare_grpo_data import parse_args as parse_prepare_args
from select_grpo_scored_subset import parse_args as parse_filter_args, passes_difficulty_filter


class GRPODataPipelineTest(unittest.TestCase):
    def test_infer_source_detects_gsm8k(self) -> None:
        self.assertEqual(infer_source("openai/gsm8k", "auto"), "gsm8k")
        self.assertEqual(infer_source("nlile/NuminaMath-1.5-RL-Verifiable", "auto"), "numinamath")

    def test_build_train_record_keeps_gsm8k_reasoning_for_grpo(self) -> None:
        row = {
            "question": "Toy question",
            "answer": "We add 2 and 3 to get 5.\n#### 5",
        }
        record = build_train_record(row, 0, "gsm8k")
        self.assertEqual(record["final_answer"], "5")
        self.assertIn("We add 2 and 3 to get 5.", str(record["solution"]))
        self.assertIn("Final answer: \\boxed{5}", str(record["solution"]))
        self.assertEqual(record["problem_type"], "Arithmetic")

    def test_passes_difficulty_filter_uses_only_score_band(self) -> None:
        row = {"sft_correct_rate": 0.4, "sft_parse_rate": 0.75}
        self.assertTrue(
            passes_difficulty_filter(
                row,
                min_correct_rate=0.25,
                max_correct_rate=0.50,
                min_parse_rate=0.50,
            )
        )

    def test_select_rows_by_budget_can_stratify(self) -> None:
        rows = [
            {"id": "1", "problem_type": "Algebra"},
            {"id": "2", "problem_type": "Algebra"},
            {"id": "3", "problem_type": "Geometry"},
            {"id": "4", "problem_type": "Geometry"},
        ]
        selected = select_rows_by_budget(rows, target_size=2, stratify_by="problem_type", seed=42)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["problem_type"] for row in selected}, {"Algebra", "Geometry"})

    def test_filter_script_accepts_optional_target_size(self) -> None:
        args = parse_filter_args(["--target-size", "100", "--output-selected", "data/out.jsonl"])
        self.assertEqual(args.target_size, 100)
        self.assertEqual(args.output_selected, "data/out.jsonl")

    def test_filter_script_defaults_to_full_filtered_output(self) -> None:
        args = parse_filter_args([])
        self.assertIsNone(args.target_size)
        self.assertIsNone(args.output_selected)

    def test_prepare_grpo_data_defaults_to_prompt_v1(self) -> None:
        args = parse_prepare_args([])
        self.assertEqual(args.prompt_version, "v1")

    def test_prepare_grpo_data_accepts_prompt_v2(self) -> None:
        args = parse_prepare_args(["--prompt-version", "v2"])
        self.assertEqual(args.prompt_version, "v2")


if __name__ == "__main__":
    unittest.main()
