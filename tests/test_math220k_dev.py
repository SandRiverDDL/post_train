from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.math220k_dev import (
    default_output_path,
    prepare_math220k_dev,
    profile_names,
    profile_spec,
)


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class Math220KDevTest(unittest.TestCase):
    def test_profile_names_and_specs(self) -> None:
        self.assertEqual(profile_names(), ["main150", "short150"])
        self.assertEqual(profile_spec("main150"), [("lt512", 0, 512, 70), ("tok512_768", 512, 768, 35), ("tok768_1380", 768, 1380, 45)])
        self.assertEqual(default_output_path("short150"), Path("data/eval/math220k_dev_short150.jsonl"))

    def test_prepare_math220k_dev_main150_uses_bucket_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "dev.jsonl"
            report_path = tmp_path / "dev.report.json"
            dataset_rows = []
            for index in range(120):
                dataset_rows.append(
                    {
                        "problem": f"短题 {index}",
                        "clean_problem": f"短题 {index}",
                        "solution": " ".join(["短"] * 100),
                        "answer": "2",
                        "problem_type": "algebra",
                        "question_type": "open",
                    }
                )
            for index in range(80):
                dataset_rows.append(
                    {
                        "problem": f"中题 {index}",
                        "clean_problem": f"中题 {index}",
                        "solution": " ".join(["中"] * 600),
                        "answer": "3",
                        "problem_type": "algebra",
                        "question_type": "open",
                    }
                )
            for index in range(70):
                dataset_rows.append(
                    {
                        "problem": f"长题 {index}",
                        "clean_problem": f"长题 {index}",
                        "solution": " ".join(["长"] * 900),
                        "answer": "4",
                        "problem_type": "algebra",
                        "question_type": "open",
                    }
                )
            dataset_rows.append(
                {
                    "problem": "选择题",
                    "clean_problem": "选择题",
                    "solution": " ".join(["选"] * 100),
                    "answer": "1",
                    "problem_type": "algebra",
                    "question_type": "MCQ",
                }
            )

            with patch("post_train.math220k_dev.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_math220k_dev(
                        profile="main150",
                        output_path=output_path,
                        report_path=report_path,
                        tokenizer_name="/tmp/model",
                    )

        report = result["report"]
        self.assertEqual(report["filtered_mcq"], 1)
        self.assertEqual(report["output_size"], 150)
        self.assertEqual(report["pools"]["lt512"]["quota"], 70)
        self.assertEqual(report["pools"]["tok512_768"]["quota"], 35)
        self.assertEqual(report["pools"]["tok768_1380"]["quota"], 45)

    def test_prepare_math220k_dev_short150_only_draws_short_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "short.jsonl"
            report_path = tmp_path / "short.report.json"
            dataset_rows = [
                {
                    "problem": f"短题 {index}",
                    "clean_problem": f"短题 {index}",
                    "solution": " ".join(["短"] * 100),
                    "answer": "2",
                    "problem_type": "algebra",
                    "question_type": "open",
                }
                for index in range(200)
            ]

            with patch("post_train.math220k_dev.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_math220k_dev(
                        profile="short150",
                        output_path=output_path,
                        report_path=report_path,
                        tokenizer_name="/tmp/model",
                    )

        report = result["report"]
        self.assertEqual(report["output_size"], 150)
        self.assertEqual(list(report["pools"].keys()), ["lt512"])
        self.assertEqual(report["pools"]["lt512"]["quota"], 150)


if __name__ == "__main__":
    unittest.main()
