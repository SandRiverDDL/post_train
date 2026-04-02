from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage2_data_config
from post_train.stage2_data import (
    assign_selection_pool,
    make_math220k_record,
    normalize_stage2_solution,
    prepare_stage2_dataset,
    validate_sft_record,
)


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class Stage2DataTest(unittest.TestCase):
    def test_load_stage2_data_config_validates_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "stage2_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "stage1_input: data/stage1/train.jsonl",
                        "tokenizer_name: /tmp/model",
                        "total_size: 10",
                        "stage1_random_quota: 2",
                        "math220k_short_quota: 6",
                        "math220k_long_quota: 2",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_data_config(config_path)

        self.assertEqual(cfg.total_size, 10)
        self.assertEqual(cfg.math220k_short_quota, 6)

    def test_make_math220k_record_prefers_clean_problem_and_filters_mcq(self) -> None:
        mcq_row = {
            "problem": "old",
            "clean_problem": "new",
            "solution": "Work",
            "answer": "2",
            "question_type": "MCQ",
        }
        self.assertIsNone(make_math220k_record(mcq_row, 0))

        row = {
            "problem": "old",
            "clean_problem": "new",
            "solution": "推导过程",
            "answer": "2",
            "problem_type": "algebra",
            "question_type": "open",
        }
        record = make_math220k_record(row, 1)

        assert record is not None
        self.assertEqual(record["question"], "new")
        self.assertTrue(record["solution"].endswith("\\boxed{2}"))
        self.assertEqual(record["final_answer"], "2")
        self.assertEqual(record["meta"]["question_type"], "open")

    def test_normalize_stage2_solution_keeps_single_valid_boxed(self) -> None:
        solution, meta = normalize_stage2_solution("推导过程\n\n\\boxed{2}", "2")
        self.assertEqual(solution, "推导过程\n\n\\boxed{2}")
        self.assertEqual(meta["action"], "already_valid_boxed")
        self.assertEqual(meta["boxed_count_after"], 1)

    def test_normalize_stage2_solution_rewrites_double_boxed_tail(self) -> None:
        raw_solution = "推导过程\nTherefore, the answer is \\(\\boxed{2}\\).\n\n\\boxed{2}"
        solution, meta = normalize_stage2_solution(raw_solution, "2")
        self.assertEqual(solution, "推导过程\n\n\\boxed{2}")
        self.assertEqual(meta["action"], "rewritten_tail")
        self.assertEqual(meta["boxed_count_before"], 2)
        self.assertEqual(meta["boxed_count_after"], 1)

    def test_normalize_stage2_solution_appends_single_boxed_when_missing(self) -> None:
        solution, meta = normalize_stage2_solution("推导过程", "2")
        self.assertEqual(solution, "推导过程\n\n\\boxed{2}")
        self.assertEqual(meta["action"], "appended_boxed")
        self.assertEqual(meta["boxed_count_after"], 1)

    def test_validate_sft_record_rejects_answer_mismatch(self) -> None:
        ok, reason = validate_sft_record(
            {
                "question": "1+1=?",
                "solution": "推导\n\n\\boxed{3}",
                "final_answer": "2",
            }
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_mismatch")

    def test_assign_selection_pool_uses_source_and_token_thresholds(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(
            assign_selection_pool(
                {"solution": "a b c", "meta": {"source": "stage1"}},
                tokenizer=tokenizer,
                short_max_tokens=5,
                long_max_tokens=10,
            ),
            "stage1_random",
        )
        self.assertEqual(
            assign_selection_pool(
                {"solution": "a b c", "meta": {"source": "math220k"}},
                tokenizer=tokenizer,
                short_max_tokens=5,
                long_max_tokens=10,
            ),
            "math220k_short",
        )
        self.assertEqual(
            assign_selection_pool(
                {"solution": "a b c d e f", "meta": {"source": "math220k"}},
                tokenizer=tokenizer,
                short_max_tokens=5,
                long_max_tokens=10,
            ),
            "math220k_long",
        )
        self.assertIsNone(
            assign_selection_pool(
                {"solution": "a b c d e f g h i j k", "meta": {"source": "math220k"}},
                tokenizer=tokenizer,
                short_max_tokens=5,
                long_max_tokens=10,
            )
        )

    def test_prepare_stage2_dataset_reports_zero_double_boxed_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            stage1_path = tmp_path / "stage1.jsonl"
            stage1_path.write_text(
                '{"id":"s1","question":"1+1=?","solution":"推导\\n\\n\\\\boxed{2}","final_answer":"2","meta":{}}\n',
                encoding="utf-8",
            )
            output_path = tmp_path / "stage2.jsonl"
            report_path = tmp_path / "stage2.report.json"
            config_path = tmp_path / "stage2_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"stage1_input: {stage1_path}",
                        "tokenizer_name: /tmp/model",
                        f"output_path: {output_path}",
                        f"report_path: {report_path}",
                        "total_size: 3",
                        "stage1_random_quota: 1",
                        "math220k_short_quota: 2",
                        "math220k_long_quota: 0",
                        "short_max_tokens: 50",
                        "long_max_tokens: 100",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_data_config(config_path)
            dataset_rows = [
                {
                    "problem": "old",
                    "clean_problem": "题目一",
                    "solution": "推导过程\nTherefore, the answer is \\(\\boxed{2}\\).\n\n\\boxed{2}",
                    "answer": "2",
                    "problem_type": "algebra",
                    "question_type": "open",
                },
                {
                    "problem": "old",
                    "clean_problem": "题目二",
                    "solution": "另一段推导",
                    "answer": "3",
                    "problem_type": "algebra",
                    "question_type": "open",
                },
            ]

            with patch("post_train.stage2_data.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_stage2_dataset(cfg)

        report = result["report"]
        self.assertEqual(report["normalization"]["double_boxed_detected_count"], 1)
        self.assertEqual(report["normalization"]["double_boxed_after_normalization_count"], 0)


if __name__ == "__main__":
    unittest.main()
