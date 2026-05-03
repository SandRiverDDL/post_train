from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage2_mix_long_data_config
from post_train.datasets.stage2_mix_long import make_mix_long_record, prepare_stage2_mix_long_dataset


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class Stage2MixLongDataTest(unittest.TestCase):
    def test_load_stage2_mix_long_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mix_long.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_name: demo/mix-long",
                        "dataset_split: train",
                        "tokenizer_name: /tmp/model",
                        "seed: 7",
                        "max_solution_tokens: 1024",
                        "sample_size: 10",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_mix_long_data_config(config_path)

        self.assertEqual(cfg.dataset_name, "demo/mix-long")
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.max_solution_tokens, 1024)
        self.assertEqual(cfg.sample_size, 10)

    def test_make_mix_long_record_keeps_existing_boxed(self) -> None:
        record = make_mix_long_record(
            {
                "id": "boxed",
                "question": "1+1=?",
                "solution": "Reasoning\n\n**Final Answer**\n\n\\[\\boxed{2}\\]",
                "final_answer": "2",
            },
            0,
            source="demo/mix-long",
        )

        self.assertEqual(record["solution"].count("\\boxed{"), 1)
        self.assertIn("\\[\\boxed{2}\\]", record["solution"])
        self.assertEqual(record["meta"]["normalization_action"], "kept_existing_boxed")

    def test_make_mix_long_record_appends_bare_boxed_only_when_missing(self) -> None:
        record = make_mix_long_record(
            {
                "id": "missing",
                "question": "1+1=?",
                "solution": "Reasoning gives 2.",
                "final_answer": "2",
            },
            0,
            source="demo/mix-long",
        )

        self.assertTrue(record["solution"].endswith("\n\n\\boxed{2}"))
        self.assertEqual(record["solution"].count("\\boxed{"), 1)
        self.assertEqual(record["meta"]["normalization_action"], "appended_missing_boxed")

    def test_prepare_stage2_mix_long_dataset_filters_too_long_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "mix_long.jsonl"
            report_path = tmp_path / "mix_long.report.json"
            config_path = tmp_path / "mix_long.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_name: demo/mix-long",
                        "dataset_split: train",
                        f"output_path: {output_path}",
                        f"report_path: {report_path}",
                        "tokenizer_name: /tmp/model",
                        "max_solution_tokens: 4",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_mix_long_data_config(config_path)
            dataset_rows = [
                {
                    "id": "short",
                    "question": "1+1=?",
                    "solution": "a b c",
                    "final_answer": "2",
                },
                {
                    "id": "long",
                    "question": "2+2=?",
                    "solution": "a b c d e",
                    "final_answer": "4",
                },
            ]

            with patch("post_train.datasets.stage2_mix_long.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_stage2_mix_long_dataset(cfg)

        report = result["report"]
        self.assertEqual(report["source"]["raw_rows"], 2)
        self.assertEqual(report["source"]["kept_rows"], 1)
        self.assertEqual(report["source"]["sampled_rows"], 1)
        self.assertEqual(report["filters"]["filtered_too_long_solution"], 1)
        self.assertEqual(report["solution_tokens"]["kept"]["max"], 4)
        self.assertEqual(report["dataset_summary"]["total"], 1)

    def test_prepare_stage2_mix_long_dataset_samples_after_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "mix_long.jsonl"
            report_path = tmp_path / "mix_long.report.json"
            config_path = tmp_path / "mix_long.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_name: demo/mix-long",
                        "dataset_split: train",
                        f"output_path: {output_path}",
                        f"report_path: {report_path}",
                        "tokenizer_name: /tmp/model",
                        "max_solution_tokens: 10",
                        "sample_size: 2",
                        "seed: 123",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_mix_long_data_config(config_path)
            dataset_rows = [
                {"id": "a", "question": "q1", "solution": "a", "final_answer": "1"},
                {"id": "b", "question": "q2", "solution": "b", "final_answer": "2"},
                {"id": "c", "question": "q3", "solution": "c", "final_answer": "3"},
            ]

            with patch("post_train.datasets.stage2_mix_long.load_dataset_rows", return_value=dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_stage2_mix_long_dataset(cfg)

        report = result["report"]
        self.assertEqual(report["source"]["kept_rows"], 3)
        self.assertEqual(report["source"]["sampled_rows"], 2)
        self.assertEqual(report["dataset_summary"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
