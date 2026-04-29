from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config, load_stage2_hendrycks_long_data_config
from post_train.datasets.stage2_hendrycks_long import prepare_stage2_hendrycks_long_dataset


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class Stage2HendrycksLongDataTest(unittest.TestCase):
    def test_repo_configs_load(self) -> None:
        data_cfg = load_stage2_hendrycks_long_data_config(ROOT / "configs/stage2/hendrycks_long_data.yaml")
        sft_cfg = load_sft_config(ROOT / "configs/stage2/hendrycks_long_sft.yaml")

        self.assertEqual(data_cfg.total_size, 2000)
        self.assertEqual(data_cfg.levels, [4, 5])
        self.assertEqual(data_cfg.max_solution_tokens, 4096)
        self.assertEqual(len(data_cfg.hendrycks_config_names), 7)
        self.assertEqual(sft_cfg.model_name, "outputs/stage1_mix_long_sft/checkpoint-300")
        self.assertEqual(sft_cfg.train_dataset, Path("data/stage2/hendrycks_long/train.jsonl"))

    def test_prepare_stage2_hendrycks_long_dataset_matches_and_builds_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            short_path = tmp_path / "short.jsonl"
            short_path.write_text(
                "\n".join(
                    [
                        '{"id":"s1","question":"Overlap  problem","solution":"短解1\\\\n\\\\n\\\\boxed{1}","final_answer":"1","meta":{}}',
                        '{"id":"s2","question":"Short only 2","solution":"短解2\\\\n\\\\n\\\\boxed{2}","final_answer":"2","meta":{}}',
                        '{"id":"s3","question":"Short only 3","solution":"短解3\\\\n\\\\n\\\\boxed{3}","final_answer":"3","meta":{}}',
                        '{"id":"s4","question":"Short only 4","solution":"短解4\\\\n\\\\n\\\\boxed{4}","final_answer":"4","meta":{}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = tmp_path / "stage2.jsonl"
            report_path = tmp_path / "stage2.report.json"
            unmatched_path = tmp_path / "unmatched.jsonl"
            config_path = tmp_path / "stage2.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"short_dataset_path: {short_path}",
                        "hendrycks_dataset: demo/hendrycks",
                        "hendrycks_split: train",
                        "hendrycks_config_names:",
                        "  - algebra",
                        "long_cot_dataset: demo/long",
                        "long_cot_split: train",
                        f"output_path: {output_path}",
                        f"report_path: {report_path}",
                        f"unmatched_preview_path: {unmatched_path}",
                        "tokenizer_name: /tmp/model",
                        "seed: 7",
                        "total_size: 3",
                        "long_ratio: 1",
                        "short_ratio: 2",
                        "levels:",
                        "  - 4",
                        "  - 5",
                        "max_solution_tokens: 5",
                        "unmatched_preview_count: 10",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_stage2_hendrycks_long_data_config(config_path)

            hendrycks_rows = [
                {"problem": "Overlap   problem", "level": "Level 5"},
                {"problem": "Only level three", "level": "Level 3"},
                {"problem": "Unmatched  sample", "level": 4},
            ]
            long_rows = [
                {"problem": "Overlap problem", "solution": "a b c", "answer": "1"},
                {"problem": "Too long sample", "solution": "a b c d e f", "answer": "9"},
            ]

            def fake_load_dataset_rows(
                dataset_name: str,
                *,
                split: str,
                config_name: str | None = None,
                cache_dir: str | None = None,
            ) -> list[dict]:
                if dataset_name == "demo/hendrycks":
                    return hendrycks_rows
                if dataset_name == "demo/long":
                    return long_rows
                raise AssertionError(f"unexpected dataset {dataset_name}")

            with patch("post_train.datasets.stage2_hendrycks_long.load_dataset_rows", side_effect=fake_load_dataset_rows):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                    result = prepare_stage2_hendrycks_long_dataset(cfg)

        report = result["report"]
        self.assertEqual(report["sources"]["hendrycks_filter"]["level_filtered_rows"], 2)
        self.assertEqual(report["sources"]["hendrycks_filter"]["matched_problem_rows"], 1)
        self.assertAlmostEqual(report["sources"]["hendrycks_filter"]["matched_problem_rate"], 0.5)
        self.assertEqual(report["sources"]["matched_long"]["available_rows"], 1)
        self.assertEqual(report["sources"]["matched_long"]["selected_rows"], 1)
        self.assertEqual(report["sources"]["short"]["selected_rows"], 2)
        self.assertEqual(report["sources"]["short"]["overlap_excluded_rows"], 1)
        self.assertEqual(report["dataset_summary"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
