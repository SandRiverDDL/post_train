from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_on_policy_data_config
from post_train.on_policy_data import (
    build_query_pool,
    build_retained_sft_dataset,
    load_query_candidates,
)


class OnPolicyDataTest(unittest.TestCase):
    def test_load_on_policy_data_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_count: 10",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)

        self.assertEqual(cfg.base_model_name, "/tmp/base")
        self.assertEqual(cfg.query_count, 10)
        self.assertEqual(cfg.responses_per_query, 4)
        self.assertEqual(cfg.max_completion_tokens, 512)
        self.assertEqual(cfg.min_retained_count, 32)

    def test_build_query_pool_excludes_exact_question_from_excluded_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exclude_path = tmp_path / "stage1_dev200.jsonl"
            exclude_path.write_text(
                '{"id":"dev-1","question":"  What is 1 + 1? ","final_answer":"2","meta":{"source":"toy"}}\n',
                encoding="utf-8",
            )
            config_path = tmp_path / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_dataset: toy",
                        "query_source: toy",
                        "query_count: 1",
                        f"exclude_paths:\n  - {exclude_path}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)
            raw_rows = [
                {"id": "0", "problem": "What is 1 + 1?", "answer": "2"},
                {"id": "1", "problem": "What is 2 + 2?", "answer": "4"},
            ]

            with patch("post_train.on_policy_data.load_dataset_rows", return_value=raw_rows):
                query_rows, report = build_query_pool(cfg)

        self.assertEqual(report["excluded_questions"], 1)
        self.assertEqual(len(query_rows), 1)
        self.assertEqual(query_rows[0]["question"], "What is 2 + 2?")

    def test_load_query_candidates_excludes_stage1_train_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            stage1_train_path = tmp_path / "stage1_train.jsonl"
            stage1_train_path.write_text(
                '{"id":"stage1-train-1","question":"Seen question?","final_answer":"2","meta":{"source":"toy"}}\n',
                encoding="utf-8",
            )
            config_path = tmp_path / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_dataset: toy",
                        "query_source: toy",
                        "query_count: 1",
                        f"exclude_paths:\n  - {stage1_train_path}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)
            raw_rows = [
                {"id": "stage1-train-1", "problem": "Seen question?", "answer": "2"},
                {"id": "unseen-1", "problem": "Unseen question?", "answer": "4"},
            ]

            with patch("post_train.on_policy_data.load_dataset_rows", return_value=raw_rows):
                query_rows, report = load_query_candidates(cfg)

        self.assertEqual(report["available_rows"], 1)
        self.assertEqual(query_rows[0]["id"], "unseen-1")

    def test_build_retained_sft_dataset_keeps_shortest_valid_response(self) -> None:
        cfg = load_on_policy_data_config("configs/on_policy_data.yaml")
        raw_samples = [
            {
                "id": "q1",
                "question": "1+1=?",
                "final_answer": "2",
                "meta": {"source": "toy"},
                "responses": [
                    {"text": "bad", "parse_ok": False, "correct": False, "output_tokens": 1},
                    {"text": "long", "parse_ok": True, "correct": True, "output_tokens": 30},
                    {"text": "short", "parse_ok": True, "correct": True, "output_tokens": 12},
                ],
            }
        ]

        retained_rows, report = build_retained_sft_dataset(raw_samples, cfg)

        self.assertEqual(len(retained_rows), 1)
        self.assertEqual(retained_rows[0]["solution"], "short")
        self.assertEqual(report["multiple_valid_candidates"], 1)
        self.assertEqual(report["filtered_parse_fail"], 1)
        self.assertEqual(report["kept"], 1)

    def test_build_retained_sft_dataset_filters_too_long_and_reports_ratio(self) -> None:
        cfg = load_on_policy_data_config("configs/on_policy_data.yaml")
        cfg = cfg.model_copy(update={"max_completion_tokens": 10, "min_retained_count": 1})
        raw_samples = [
            {
                "id": "q1",
                "question": "1+1=?",
                "final_answer": "2",
                "meta": {"source": "toy"},
                "responses": [
                    {"text": "long", "parse_ok": True, "correct": True, "output_tokens": 20},
                ],
            },
            {
                "id": "q2",
                "question": "2+2=?",
                "final_answer": "4",
                "meta": {"source": "toy"},
                "responses": [
                    {"text": "ok", "parse_ok": True, "correct": True, "output_tokens": 8},
                ],
            },
        ]

        retained_rows, report = build_retained_sft_dataset(raw_samples, cfg)

        self.assertEqual(len(retained_rows), 1)
        self.assertEqual(report["filtered_too_long"], 1)
        self.assertAlmostEqual(report["retained_ratio"], 0.5)
        self.assertEqual(report["p50_kept_output_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
