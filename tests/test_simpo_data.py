from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_simpo_data_config
from post_train.simpo_data import build_pilot_pairs, build_preference_pairs, build_query_pool


class SimPODataTest(unittest.TestCase):
    def test_load_simpo_data_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "simpo_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_count: 10",
                        "responses_per_query: 4",
                        "target_pair_count: 5",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_simpo_data_config(config_path)

        self.assertEqual(cfg.base_model_name, "/tmp/base")
        self.assertEqual(cfg.query_count, 10)
        self.assertEqual(cfg.target_pair_count, 5)
        self.assertEqual(cfg.pilot_pair_count, 150)

    def test_build_query_pool_excludes_same_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exclude_path = tmp_path / "stage1_train.jsonl"
            exclude_path.write_text(
                '{"id":"1","question":"q","final_answer":"2","meta":{"source":"UWNSL/MATH_training_split_short_cot"}}\n'
                '{"id":"999","question":"q","final_answer":"2","meta":{"source":"math500"}}\n',
                encoding="utf-8",
            )
            config_path = tmp_path / "simpo_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "query_count: 2",
                        f"exclude_paths:\n  - {exclude_path}",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_simpo_data_config(config_path)
            raw_rows = [
                {"id": "0", "problem": "q0", "answer": "0"},
                {"id": "1", "problem": "q1", "answer": "1"},
                {"id": "2", "problem": "q2", "answer": "2"},
            ]

            with patch("post_train.simpo_data.load_dataset_rows", return_value=raw_rows):
                query_rows, report = build_query_pool(cfg)

        self.assertEqual(report["excluded_ids"], 1)
        self.assertEqual(len(query_rows), 2)
        self.assertNotIn("1", {row["id"] for row in query_rows})

    def test_build_preference_pairs_prefers_correct_vs_incorrect(self) -> None:
        cfg = load_simpo_data_config("configs/simpo_data.yaml")
        raw_samples = [
            {
                "id": "q1",
                "prompt": "prompt",
                "responses": [
                    {"text": "short correct", "boxed": True, "parse_ok": True, "correct": True, "output_tokens": 10},
                    {"text": "long correct", "boxed": True, "parse_ok": True, "correct": True, "output_tokens": 30},
                    {"text": "wrong", "boxed": False, "parse_ok": False, "correct": False, "output_tokens": 50},
                ],
            }
        ]
        pairs, report = build_preference_pairs(raw_samples, cfg)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(report["correct_vs_incorrect_pairs"], 1)
        self.assertEqual(pairs[0]["meta"]["pair_type"], "correct_vs_incorrect")
        self.assertEqual(pairs[0]["chosen"], "short correct")
        self.assertEqual(pairs[0]["rejected"], "wrong")

    def test_build_preference_pairs_falls_back_to_correct_vs_correct_with_gap(self) -> None:
        cfg = load_simpo_data_config("configs/simpo_data.yaml")
        raw_samples = [
            {
                "id": "q2",
                "prompt": "prompt",
                "responses": [
                    {"text": "tight", "boxed": True, "parse_ok": True, "correct": True, "output_tokens": 20},
                    {"text": "verbose answer", "boxed": True, "parse_ok": True, "correct": True, "output_tokens": 80},
                ],
            }
        ]
        pairs, report = build_preference_pairs(raw_samples, cfg)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(report["correct_vs_correct_pairs"], 1)
        self.assertEqual(pairs[0]["meta"]["pair_type"], "correct_vs_correct")
        self.assertEqual(pairs[0]["chosen"], "tight")
        self.assertEqual(pairs[0]["rejected"], "verbose answer")

    def test_build_pilot_pairs_prefers_correct_vs_incorrect(self) -> None:
        cfg = load_simpo_data_config("configs/simpo_data.yaml")
        cfg = cfg.model_copy(update={"pilot_pair_count": 2, "seed": 7})
        pair_rows = [
            {
                "id": "a",
                "prompt": "p",
                "chosen": "c1",
                "rejected": "r1",
                "meta": {"pair_type": "correct_vs_incorrect"},
            },
            {
                "id": "b",
                "prompt": "p",
                "chosen": "c2",
                "rejected": "r2",
                "meta": {"pair_type": "correct_vs_incorrect"},
            },
            {
                "id": "c",
                "prompt": "p",
                "chosen": "c3",
                "rejected": "r3",
                "meta": {"pair_type": "correct_vs_correct"},
            },
        ]

        pilot_rows, report = build_pilot_pairs(pair_rows, cfg)

        self.assertEqual(len(pilot_rows), 2)
        self.assertEqual(report["correct_vs_incorrect_pairs"], 2)
        self.assertEqual(report["correct_vs_correct_pairs"], 0)
        self.assertTrue(
            all(row["meta"]["pair_type"] == "correct_vs_incorrect" for row in pilot_rows)
        )

    def test_build_pilot_pairs_backfills_with_correct_vs_correct(self) -> None:
        cfg = load_simpo_data_config("configs/simpo_data.yaml")
        cfg = cfg.model_copy(update={"pilot_pair_count": 3, "seed": 7})
        pair_rows = [
            {
                "id": "a",
                "prompt": "p",
                "chosen": "c1",
                "rejected": "r1",
                "meta": {"pair_type": "correct_vs_incorrect"},
            },
            {
                "id": "b",
                "prompt": "p",
                "chosen": "c2",
                "rejected": "r2",
                "meta": {"pair_type": "correct_vs_correct"},
            },
            {
                "id": "c",
                "prompt": "p",
                "chosen": "c3",
                "rejected": "r3",
                "meta": {"pair_type": "correct_vs_correct"},
            },
        ]

        pilot_rows, report = build_pilot_pairs(pair_rows, cfg)

        self.assertEqual(len(pilot_rows), 3)
        self.assertEqual(report["correct_vs_incorrect_pairs"], 1)
        self.assertEqual(report["correct_vs_correct_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
