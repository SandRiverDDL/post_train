from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import (
    Stage1RSRCandidatesConfig,
    Stage1RSRSelectConfig,
    load_stage1_rsr_candidates_config,
    load_stage1_rsr_select_config,
)
from post_train.rsr_data import (
    _build_rsr_diagnostic_line,
    _compute_rsr_from_logits,
    _group_rows_by_length,
    prepare_stage1_rsr_candidates,
    select_stage1_rsr_dataset,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        pieces = [piece for piece in text.split() if piece]
        return {"input_ids": list(range(1, len(pieces) + 1))}


class RSRMetricTest(unittest.TestCase):
    def test_compute_rsr_from_logits_matches_expected_single_token_case(self) -> None:
        logits = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
        input_ids = [1, 1]
        completion_positions = [1]

        metrics = _compute_rsr_from_logits(
            logits,
            input_ids,
            completion_positions,
            rank_clip_r=100,
        )

        expected_nll = torch.logsumexp(logits[0], dim=-1).item() - 2.0
        self.assertEqual(metrics["resp_token_length"], 1)
        self.assertAlmostEqual(metrics["avg_surprisal"], expected_nll, places=6)
        self.assertAlmostEqual(metrics["avg_rank_clip"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rank_surprisal_ratio"], 1.0 / expected_nll, places=6)

    def test_group_rows_by_length_sorts_longer_sequences_first(self) -> None:
        rows = [
            {"id": "b", "input_ids": [1, 2]},
            {"id": "a", "input_ids": [1, 2, 3, 4]},
            {"id": "c", "input_ids": [1, 2, 3]},
        ]

        grouped = _group_rows_by_length(rows)

        self.assertEqual([row["id"] for row in grouped], ["a", "c", "b"])

    def test_build_rsr_diagnostic_line_handles_missing_cuda_metrics(self) -> None:
        line = _build_rsr_diagnostic_line(
            batch_index=10,
            total_batches=100,
            batch_lengths=[2560, 2400],
            allocated_bytes=None,
            reserved_bytes=None,
            peak_allocated_bytes=None,
        )

        self.assertIn("RSR batch 10/100", line)
        self.assertIn("len[min/avg/max]=2400/2480.0/2560", line)
        self.assertIn("alloc=n/a", line)
        self.assertIn("reserved=n/a", line)


class RSRCandidatesTest(unittest.TestCase):
    def test_prepare_stage1_rsr_candidates_builds_coverage_buckets_and_best_source(self) -> None:
        source_rows = {
            "UWNSL/MATH_training_split_short_cot": [
                {"problem": "A", "solution": "short a\n\n\\boxed{1}"},
                {"problem": "B", "solution": "short b\n\n\\boxed{2}"},
            ],
            "UWNSL/MATH_training_split_long_cot": [
                {"problem": "A", "solution": "long a\n\n\\boxed{1}"},
                {"problem": "B", "solution": "long b\n\n\\boxed{2}"},
            ],
            "UWNSL/MATH_training_split_distill_small_teacher": [
                {"problem": "A", "solution": "small a\n\n\\boxed{1}"},
                {"problem": "C", "solution": "small c\n\n\\boxed{3}"},
            ],
            "UWNSL/MATH_training_split_distill_large_teacher": [
                {"problem": "A", "solution": "large a\n\n\\boxed{1}"},
                {"problem": "C", "solution": "large c\n\n\\boxed{3}"},
            ],
        }

        def fake_load_dataset_rows(dataset_name: str, *, split: str, cache_dir: str | None = None):
            return list(source_rows[dataset_name])

        def fake_score(rows, cfg, *, tokenizer=None):
            score_map = {"large": 4.0, "small": 3.0, "short": 2.0, "long": 1.0}
            scored = []
            for row in rows:
                copied = dict(row)
                copied["prompt_tokens"] = 10
                copied["solution_tokens"] = int(row["meta"]["solution_tokens"])
                copied["scored_solution_tokens"] = int(row["meta"]["solution_tokens"])
                copied["truncated"] = False
                copied["avg_rank_clip"] = 10.0
                copied["avg_surprisal"] = 2.0
                copied["rank_surprisal_ratio"] = score_map[row["meta"]["source_name"]]
                scored.append(copied)
            return scored

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "post_train.rsr_data.load_dataset_rows",
            side_effect=fake_load_dataset_rows,
        ), patch(
            "post_train.rsr_data.score_rsr_trajectories",
            side_effect=fake_score,
        ):
            cfg = Stage1RSRCandidatesConfig(
                student_model_name="demo/student",
                tokenizer_name="demo/tokenizer",
                output_path=Path(tmp_dir) / "candidates.jsonl",
                report_path=Path(tmp_dir) / "candidates.report.json",
                unmatched_preview_path=Path(tmp_dir) / "candidates.unmatched.jsonl",
            )
            result = prepare_stage1_rsr_candidates(cfg, tokenizer=DummyTokenizer())
            self.assertEqual(result["report"]["coverage"]["trajectory_count_distribution"], {2: 2, 4: 1})
            self.assertEqual(result["report"]["coverage"]["source_combo_distribution"]["large+small+short+long"], 1)
            written_rows = [json.loads(line) for line in Path(result["output_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(written_rows), 3)
            by_question = {row["question"]: row for row in written_rows}
            self.assertEqual(by_question["A"]["trajectory_count"], 4)
            self.assertEqual(by_question["A"]["best_source_by_rsr"], "long")
            self.assertEqual(by_question["B"]["trajectory_count"], 2)
            self.assertEqual(by_question["C"]["trajectory_count"], 2)

    def test_prepare_stage1_rsr_candidates_drops_truncated_rows_by_default(self) -> None:
        source_rows = {
            "UWNSL/MATH_training_split_short_cot": [
                {"problem": "A", "solution": "x " * 10 + "\\boxed{1}"},
            ],
            "UWNSL/MATH_training_split_long_cot": [
                {"problem": "A", "solution": "x " * 10 + "\\boxed{1}"},
            ],
            "UWNSL/MATH_training_split_distill_small_teacher": [],
            "UWNSL/MATH_training_split_distill_large_teacher": [],
        }

        def fake_load_dataset_rows(dataset_name: str, *, split: str, cache_dir: str | None = None):
            return list(source_rows[dataset_name])

        def fake_score(rows, cfg, *, tokenizer=None):
            self.assertEqual(rows, [])
            return []

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "post_train.rsr_data.load_dataset_rows",
            side_effect=fake_load_dataset_rows,
        ), patch(
            "post_train.rsr_data.score_rsr_trajectories",
            side_effect=fake_score,
        ):
            cfg = Stage1RSRCandidatesConfig(
                student_model_name="demo/student",
                tokenizer_name="demo/tokenizer",
                output_path=Path(tmp_dir) / "candidates.jsonl",
                report_path=Path(tmp_dir) / "candidates.report.json",
                unmatched_preview_path=Path(tmp_dir) / "candidates.unmatched.jsonl",
                max_seq_length=8,
                drop_truncated=True,
            )
            result = prepare_stage1_rsr_candidates(cfg, tokenizer=DummyTokenizer())
            self.assertEqual(result["report"]["filters"]["truncated_before_scoring"], 2)
            self.assertEqual(result["report"]["filters"]["dropped_truncated"], 2)
            self.assertEqual(result["report"]["filters"]["dropped_truncated_ratio"], 1.0)


class RSRSelectTest(unittest.TestCase):
    def test_select_stage1_rsr_dataset_prioritizes_higher_trajectory_count_and_lower_rsr(self) -> None:
        candidate_rows = [
            {
                "id": "a",
                "question": "A",
                "trajectory_count": 4,
                "trajectories": [
                    {
                        "source_name": "large",
                        "source_dataset": "large_ds",
                        "solution": "sol a1\n\n\\boxed{1}",
                        "final_answer": "1",
                        "solution_tokens": 100,
                        "rank_surprisal_ratio": 1.0,
                    },
                    {
                        "source_name": "short",
                        "source_dataset": "short_ds",
                        "solution": "sol a2\n\n\\boxed{1}",
                        "final_answer": "1",
                        "solution_tokens": 120,
                        "rank_surprisal_ratio": 0.9,
                    },
                ],
            },
            {
                "id": "b",
                "question": "B",
                "trajectory_count": 2,
                "trajectories": [
                    {
                        "source_name": "large",
                        "source_dataset": "large_ds",
                        "solution": "sol b1\n\n\\boxed{2}",
                        "final_answer": "2",
                        "solution_tokens": 900,
                        "rank_surprisal_ratio": 1.2,
                    },
                    {
                        "source_name": "short",
                        "source_dataset": "short_ds",
                        "solution": "sol b2\n\n\\boxed{2}",
                        "final_answer": "2",
                        "solution_tokens": 140,
                        "rank_surprisal_ratio": 1.1,
                    },
                ],
            },
            {
                "id": "c",
                "question": "C",
                "trajectory_count": 2,
                "trajectories": [
                    {
                        "source_name": "small",
                        "source_dataset": "small_ds",
                        "solution": "sol c1\n\n\\boxed{3}",
                        "final_answer": "3",
                        "solution_tokens": 150,
                        "rank_surprisal_ratio": 1.3,
                    }
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "candidates.jsonl"
            with input_path.open("w", encoding="utf-8") as fh:
                for row in candidate_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            cfg = Stage1RSRSelectConfig(
                input_path=input_path,
                output_path=Path(tmp_dir) / "selected.jsonl",
                report_path=Path(tmp_dir) / "selected.report.json",
                sample_size=2,
                min_solution_tokens=0,
                max_solution_tokens=800,
                drop_truncated_trajectories=True,
                prefer_lower_rsr=True,
            )
            result = select_stage1_rsr_dataset(cfg)
            selected_rows = [json.loads(line) for line in Path(result["output_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in selected_rows], ["a", "b"])
            self.assertEqual(selected_rows[0]["meta"]["source_name"], "short")
            self.assertEqual(selected_rows[1]["meta"]["source_name"], "short")
            self.assertEqual(result["report"]["selection"]["chosen_source_counts"], {"short": 2})

    def test_select_stage1_rsr_dataset_respects_allowed_sources(self) -> None:
        candidate_rows = [
            {
                "id": "a",
                "question": "A",
                "trajectory_count": 4,
                "trajectories": [
                    {
                        "source_name": "long",
                        "source_dataset": "long_ds",
                        "solution": "sol long\n\n\\boxed{1}",
                        "final_answer": "1",
                        "solution_tokens": 300,
                        "rank_surprisal_ratio": 0.5,
                        "truncated": False,
                    },
                    {
                        "source_name": "short",
                        "source_dataset": "short_ds",
                        "solution": "sol short\n\n\\boxed{1}",
                        "final_answer": "1",
                        "solution_tokens": 120,
                        "rank_surprisal_ratio": 0.9,
                        "truncated": False,
                    },
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "candidates.jsonl"
            with input_path.open("w", encoding="utf-8") as fh:
                for row in candidate_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            cfg = Stage1RSRSelectConfig(
                input_path=input_path,
                output_path=Path(tmp_dir) / "selected.jsonl",
                report_path=Path(tmp_dir) / "selected.report.json",
                sample_size=1,
                allowed_sources=["short", "small", "large"],
                drop_truncated_trajectories=True,
                prefer_lower_rsr=True,
            )
            result = select_stage1_rsr_dataset(cfg)
            selected_rows = [json.loads(line) for line in Path(result["output_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(selected_rows[0]["meta"]["source_name"], "short")
            self.assertEqual(result["report"]["config"]["allowed_sources"], ["large", "short", "small"])


class RSRConfigTest(unittest.TestCase):
    def test_load_rsr_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            candidates_cfg = Path(tmp_dir) / "rsr_candidates.yaml"
            candidates_cfg.write_text(
                "\n".join(
                    [
                        "student_model_name: demo/model",
                        "tokenizer_name: demo/tokenizer",
                    ]
                ),
                encoding="utf-8",
            )
            select_cfg = Path(tmp_dir) / "rsr_select.yaml"
            select_cfg.write_text(
                "\n".join(
                    [
                        "sample_size: 10",
                        "min_solution_tokens: 5",
                        "max_solution_tokens: 100",
                        "allowed_sources:",
                        "  - short",
                        "  - large",
                    ]
                ),
                encoding="utf-8",
            )

            loaded_candidates = load_stage1_rsr_candidates_config(candidates_cfg)
            loaded_select = load_stage1_rsr_select_config(select_cfg)

        self.assertEqual(loaded_candidates.student_model_name, "demo/model")
        self.assertEqual(loaded_candidates.tokenizer_name, "demo/tokenizer")
        self.assertTrue(loaded_candidates.drop_truncated)
        self.assertEqual(loaded_select.sample_size, 10)
        self.assertEqual(loaded_select.max_solution_tokens, 100)
        self.assertEqual(loaded_select.allowed_sources, ["short", "large"])
        self.assertTrue(loaded_select.drop_truncated_trajectories)
        self.assertTrue(loaded_select.prefer_lower_rsr)


if __name__ == "__main__":
    unittest.main()
