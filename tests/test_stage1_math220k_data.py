from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage1_math220k_data_config, load_sft_config
from post_train.stage1_math220k_data import prepare_stage1_math220k_dataset


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class Stage1Math220kDataTest(unittest.TestCase):
    def test_load_stage1_math220k_data_config_validates_token_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "stage1_math220k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "tokenizer_name: /tmp/model",
                        "sample_size: 10",
                        "min_solution_tokens: 5",
                        "max_solution_tokens: 20",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage1_math220k_data_config(config_path)

        self.assertEqual(cfg.sample_size, 10)
        self.assertEqual(cfg.min_solution_tokens, 5)
        self.assertEqual(cfg.max_solution_tokens, 20)

    def test_prepare_stage1_math220k_dataset_filters_and_samples_by_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "stage1_math220k.yaml"
            output_path = tmp_path / "train.jsonl"
            report_path = tmp_path / "train.report.json"
            config_path.write_text(
                "\n".join(
                    [
                        "tokenizer_name: /tmp/model",
                        f"output_path: {output_path}",
                        f"report_path: {report_path}",
                        "sample_size: 2",
                        "min_solution_tokens: 3",
                        "max_solution_tokens: 6",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage1_math220k_data_config(config_path)
            dataset_rows = [
                {
                    "clean_problem": "题目一",
                    "solution": "短",
                    "answer": "1",
                    "question_type": "open",
                },
                {
                    "clean_problem": "题目二",
                    "solution": "一 二 三",
                    "answer": "2",
                    "question_type": "open",
                },
                {
                    "clean_problem": "题目三",
                    "solution": "一 二 三 四 五 六 七",
                    "answer": "3",
                    "question_type": "open",
                },
                {
                    "clean_problem": "题目四",
                    "solution": "一 二 三 四",
                    "answer": "4",
                    "question_type": "open",
                },
                {
                    "clean_problem": "题目五",
                    "solution": "一 二",
                    "answer": "5",
                    "question_type": "MCQ",
                },
            ]

            with patch("post_train.datasets.stage1_math220k.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_stage1_math220k_dataset(cfg)

        report = result["report"]
        self.assertEqual(report["sources"]["math220k"]["raw_rows"], 5)
        self.assertEqual(report["sources"]["math220k"]["filtered_mcq"], 1)
        self.assertEqual(report["filters"]["too_short"], 1)
        self.assertEqual(report["filters"]["too_long"], 1)
        self.assertEqual(report["sources"]["math220k"]["eligible_rows"], 2)
        self.assertEqual(report["sources"]["math220k"]["selected_rows"], 2)
        self.assertEqual(report["dataset_summary"]["total"], 2)

    def test_prepare_stage1_math220k_dataset_raises_when_eligible_rows_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "stage1_math220k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "tokenizer_name: /tmp/model",
                        "sample_size: 2",
                        "min_solution_tokens: 3",
                        "max_solution_tokens: 6",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage1_math220k_data_config(config_path)
            dataset_rows = [
                {
                    "clean_problem": "题目一",
                    "solution": "短",
                    "answer": "1",
                    "question_type": "open",
                }
            ]

            with patch("post_train.datasets.stage1_math220k.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    with self.assertRaisesRegex(ValueError, "样本不足"):
                        prepare_stage1_math220k_dataset(cfg)

    def test_stage1_math220k_yaml_configs_load(self) -> None:
        data_cfg = load_stage1_math220k_data_config(ROOT / "configs/stage1/math220k_data.yaml")
        self.assertEqual(data_cfg.dataset_name, "qingy2024/OpenR1-Math-220k-Cleaned")
        self.assertGreater(data_cfg.sample_size, 0)
        self.assertGreater(data_cfg.max_solution_tokens, data_cfg.min_solution_tokens)

        sft_cfg = load_sft_config(ROOT / "configs/stage1/math220k_sft.yaml")
        self.assertEqual(sft_cfg.train_dataset, Path("data/stage1/math220k/train.jsonl"))
        self.assertEqual(sft_cfg.output_dir, Path("outputs/stage1_math220k_sft"))
        self.assertFalse(sft_cfg.profit_enabled)


if __name__ == "__main__":
    unittest.main()
