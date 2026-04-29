from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_on_policy_data_config
from post_train.io import read_jsonl
from post_train.on_policy_data import (
    SELECTOR_MIXED_ONLY,
    build_query_pool,
    build_retained_sft_dataset,
    load_query_candidates,
    prepare_on_policy_sft_dataset_from_queries,
)


class OnPolicyDataTest(unittest.TestCase):
    def test_load_on_policy_data_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)

        self.assertEqual(cfg.base_model_name, "/tmp/base")
        self.assertIsNone(cfg.query_limit)
        self.assertEqual(cfg.responses_per_query, 4)
        self.assertEqual(cfg.max_completion_tokens, 512)
        self.assertEqual(cfg.min_retained_count, 32)

    def test_load_on_policy_data_config_supports_legacy_query_count(self) -> None:
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

        self.assertEqual(cfg.query_limit, 10)

    def test_load_query_candidates_prefers_local_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            query_file = tmp_path / "queries.jsonl"
            query_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "q1",
                                "question": "What is 2 + 2?",
                                "final_answer": "4",
                                "meta": {"source": "toy-file"},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "q2",
                                "question": "What is 3 + 3?",
                                "final_answer": "6",
                                "meta": {"source": "toy-file"},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        f"query_file_path: {query_file}",
                        "query_source: toy-file",
                        "query_limit: 2",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)

            with patch("post_train.on_policy.data.load_dataset_rows") as mock_loader:
                query_rows, report = load_query_candidates(cfg)

        mock_loader.assert_not_called()
        self.assertEqual(report["available_rows"], 2)
        self.assertEqual([row["id"] for row in query_rows], ["q1", "q2"])

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
                        "query_limit: 1",
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

            with patch("post_train.on_policy.data.load_dataset_rows", return_value=raw_rows):
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
                        "query_limit: 1",
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

            with patch("post_train.on_policy.data.load_dataset_rows", return_value=raw_rows):
                query_rows, report = load_query_candidates(cfg)

        self.assertEqual(report["available_rows"], 1)
        self.assertEqual(query_rows[0]["id"], "unseen-1")

    def test_build_query_pool_without_query_limit_uses_full_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_dataset: toy",
                        "query_source: toy",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)
            raw_rows = [
                {"id": "0", "problem": "What is 1 + 1?", "answer": "2"},
                {"id": "1", "problem": "What is 2 + 2?", "answer": "4"},
            ]

            with patch("post_train.on_policy.data.load_dataset_rows", return_value=raw_rows):
                query_rows, report = build_query_pool(cfg)

        self.assertEqual(len(query_rows), 2)
        self.assertEqual(report["registry_size"], 2)
        self.assertEqual(report["effective_query_count"], 2)

    def test_build_retained_sft_dataset_keeps_shortest_valid_response(self) -> None:
        cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
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

    def test_build_retained_sft_dataset_mixed_selector_keeps_only_mixed_questions(self) -> None:
        cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
        raw_samples = [
            {
                "id": "mixed",
                "question": "1+1=?",
                "final_answer": "2",
                "meta": {"source": "toy"},
                "responses": [
                    {"text": "wrong", "parse_ok": True, "correct": False, "output_tokens": 3},
                    {"text": "short", "parse_ok": True, "correct": True, "output_tokens": 5},
                    {"text": "long", "parse_ok": True, "correct": True, "output_tokens": 8},
                ],
            },
            {
                "id": "all-correct",
                "question": "2+2=?",
                "final_answer": "4",
                "meta": {"source": "toy"},
                "responses": [
                    {"text": "ok-1", "parse_ok": True, "correct": True, "output_tokens": 3},
                    {"text": "ok-2", "parse_ok": True, "correct": True, "output_tokens": 4},
                ],
            },
        ]

        retained_rows, report = build_retained_sft_dataset(
            raw_samples,
            cfg,
            selector_name=SELECTOR_MIXED_ONLY,
        )

        self.assertEqual(len(retained_rows), 1)
        self.assertEqual(retained_rows[0]["id"], "mixed")
        self.assertEqual(retained_rows[0]["solution"], "short")
        self.assertEqual(report["questions_with_mixed_outcome"], 1)
        self.assertEqual(report["questions_without_valid"], 1)

    def test_build_retained_sft_dataset_filters_too_long_and_reports_ratio(self) -> None:
        cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
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

    def test_prepare_on_policy_dataset_writes_dual_selector_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "on_policy_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_source: toy",
                        "primary_selector: any_correct_shortest",
                        "min_retained_count: 1",
                        f"query_output_path: {tmp_path / 'query.jsonl'}",
                        f"raw_samples_output_path: {tmp_path / 'raw.jsonl'}",
                        f"retained_output_path: {tmp_path / 'train.jsonl'}",
                        f"report_path: {tmp_path / 'report.json'}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_on_policy_data_config(config_path)
            query_rows = [
                {"id": "q1", "question": "1+1=?", "final_answer": "2", "meta": {"source": "toy"}},
                {"id": "q2", "question": "2+2=?", "final_answer": "4", "meta": {"source": "toy"}},
            ]
            raw_samples = [
                {
                    "id": "q1",
                    "question": "1+1=?",
                    "final_answer": "2",
                    "prompt": "prompt-1",
                    "meta": {"source": "toy"},
                    "responses": [
                        {"text": "wrong", "parse_ok": True, "correct": False, "output_tokens": 2},
                        {"text": "short", "parse_ok": True, "correct": True, "output_tokens": 3},
                    ],
                },
                {
                    "id": "q2",
                    "question": "2+2=?",
                    "final_answer": "4",
                    "prompt": "prompt-2",
                    "meta": {"source": "toy"},
                    "responses": [
                        {"text": "ok-short", "parse_ok": True, "correct": True, "output_tokens": 2},
                        {"text": "ok-long", "parse_ok": True, "correct": True, "output_tokens": 5},
                    ],
                },
            ]

            with patch(
                "post_train.on_policy.data.sample_candidate_responses",
                return_value=(raw_samples, {"sample_count": 2}),
            ):
                result = prepare_on_policy_sft_dataset_from_queries(
                    cfg,
                    query_rows=query_rows,
                    query_report={"sampled_queries": 2, "effective_query_count": 2},
                )

            primary_rows = read_jsonl(result["retained_output_path"])
            mixed_rows = read_jsonl(result["retained_output_paths"]["mixed_only_shortest"])

        self.assertEqual(len(primary_rows), 2)
        self.assertEqual(len(mixed_rows), 1)
        self.assertEqual(mixed_rows[0]["id"], "q1")
        self.assertEqual(result["report"]["raw_samples"]["mixed"], 1)
        self.assertEqual(result["report"]["raw_samples"]["all_correct"], 1)


if __name__ == "__main__":
    unittest.main()
