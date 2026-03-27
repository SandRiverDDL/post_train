from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import (
    load_on_policy_data_config,
    load_on_policy_loop_config,
    load_sft_config,
)
from post_train.on_policy_data import InsufficientRetainedSamplesError
from post_train.on_policy_loop import (
    build_query_sampler_state,
    sample_round_queries,
    build_round_data_config,
    build_round_name,
    build_round_paths,
    build_round_train_config,
    run_on_policy_loop,
)


class OnPolicyLoopTest(unittest.TestCase):
    def test_load_on_policy_loop_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "loop.yaml"
            config_path.write_text("", encoding="utf-8")
            cfg = load_on_policy_loop_config(config_path)

        self.assertEqual(cfg.seed_model, "outputs/stage1_sft/checkpoint-50")
        self.assertEqual(cfg.max_rounds, 10)
        self.assertEqual(cfg.patience, 3)
        self.assertEqual(cfg.min_delta, 0.0)

    def test_build_round_paths_uses_round_index(self) -> None:
        cfg = load_on_policy_loop_config("configs/on_policy_loop.yaml")
        paths = build_round_paths(cfg, 2)

        self.assertEqual(paths["data_dir"], Path("data/on_policy_loop/round2"))
        self.assertEqual(paths["output_dir"], Path("outputs/on_policy_loop/round2"))

    def test_build_round_configs_override_round_specific_fields(self) -> None:
        loop_cfg = load_on_policy_loop_config("configs/on_policy_loop.yaml")
        data_cfg = build_round_data_config(
            load_on_policy_data_config("configs/on_policy_data.yaml"),
            loop_cfg,
            round_index=3,
            generation_model="model-x",
        )
        base_train_cfg = load_sft_config("configs/on_policy_sft.yaml")
        train_cfg = build_round_train_config(
            base_train_cfg,
            loop_cfg,
            round_index=3,
            model_name="model-y",
            train_dataset="data/custom.jsonl",
        )

        self.assertEqual(data_cfg.round_name, "round3")
        self.assertEqual(data_cfg.generation_model, "model-x")
        self.assertEqual(data_cfg.retained_output_path, Path("data/on_policy_loop/round3/train.jsonl"))
        self.assertEqual(train_cfg.model_name, "model-y")
        self.assertEqual(train_cfg.output_dir, Path("outputs/on_policy_loop/round3"))
        self.assertEqual(train_cfg.save_strategy, "no")
        self.assertIsNone(train_cfg.save_steps)
        self.assertIsNone(train_cfg.save_total_limit)

    def test_sample_round_queries_consumes_epoch_without_cross_round_repeat(self) -> None:
        query_rows = [{"id": f"q{i}"} for i in range(6)]
        sampler_state = build_query_sampler_state(query_rows, seed=42)

        round1, report1 = sample_round_queries(query_rows, sampler_state, requested_count=2)
        round2, report2 = sample_round_queries(query_rows, sampler_state, requested_count=2)
        round3, report3 = sample_round_queries(query_rows, sampler_state, requested_count=2)

        ids1 = {row["id"] for row in round1}
        ids2 = {row["id"] for row in round2}
        ids3 = {row["id"] for row in round3}
        self.assertEqual(len(ids1 & ids2), 0)
        self.assertEqual(len(ids1 & ids3), 0)
        self.assertEqual(len(ids2 & ids3), 0)
        self.assertFalse(report1["crossed_epoch"])
        self.assertFalse(report2["crossed_epoch"])
        self.assertFalse(report3["crossed_epoch"])

    def test_sample_round_queries_reshuffles_after_epoch_exhausted(self) -> None:
        query_rows = [{"id": f"q{i}"} for i in range(5)]
        sampler_state = build_query_sampler_state(query_rows, seed=7)

        _, _ = sample_round_queries(query_rows, sampler_state, requested_count=3)
        round2, report2 = sample_round_queries(query_rows, sampler_state, requested_count=3)

        self.assertEqual(len(round2), 3)
        self.assertTrue(report2["crossed_epoch"])
        self.assertEqual(report2["effective_query_count"], 3)
        self.assertGreaterEqual(report2["epoch_index_end"], report2["epoch_index_start"])

    def test_sample_round_queries_caps_query_count_by_pool_size(self) -> None:
        query_rows = [{"id": f"q{i}"} for i in range(3)]
        sampler_state = build_query_sampler_state(query_rows, seed=1)

        round_rows, report = sample_round_queries(query_rows, sampler_state, requested_count=8)

        self.assertEqual(len(round_rows), 3)
        self.assertEqual(len({row["id"] for row in round_rows}), 3)
        self.assertEqual(report["effective_query_count"], 3)

    def test_run_on_policy_loop_stops_after_patience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            loop_cfg = load_on_policy_loop_config("configs/on_policy_loop.yaml").model_copy(
                update={
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                    "max_rounds": 5,
                    "patience": 3,
                    "seed_model": "seed-model",
                }
            )
            data_result = {
                "retained_output_path": str(tmp_path / "data/round/train.jsonl"),
                "report_path": str(tmp_path / "data/round/train.report.json"),
                "report": {"retained": {"kept": 100, "retained_ratio": 0.5}},
            }
            holdout_scores = [0.70, 0.70, 0.69, 0.69]
            data_cfg = load_on_policy_data_config("configs/on_policy_data.yaml")
            loop_queries = [
                {"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i), "meta": {"source": "toy"}}
                for i in range(12)
            ]

            def fake_holdout(**kwargs):
                score = holdout_scores.pop(0)
                return {
                    "result": {"metrics": {"normalized_accuracy": score}},
                    "result_path": str(Path(kwargs["output_dir"]) / "holdout.vllm_raw.json"),
                    "raw_result_path": str(Path(kwargs["output_dir"]) / "holdout.vllm_raw.raw.json"),
                }

            with patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=data_cfg), \
                patch("post_train.on_policy_loop.load_sft_config", return_value=load_sft_config("configs/on_policy_sft.yaml")), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.load_query_candidates", return_value=(loop_queries, {"available_rows": 12})), \
                patch("post_train.on_policy_loop.prepare_on_policy_sft_dataset_from_queries", return_value=data_result), \
                patch("post_train.on_policy_loop.train_sft") as mock_train, \
                patch("post_train.on_policy_loop.evaluate_single_dataset", side_effect=fake_holdout):
                mock_train.side_effect = lambda cfg: Path(cfg.output_dir)
                result = run_on_policy_loop(loop_cfg)

        self.assertEqual(result["summary"]["stop_reason"], "no_improvement")
        self.assertEqual(result["summary"]["completed_rounds"], 4)
        self.assertEqual(result["summary"]["best_round_index"], 1)
        self.assertEqual(result["summary"]["best_model_path"], str(tmp_path / "outputs/round1"))

    def test_run_on_policy_loop_stops_on_insufficient_retained_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            loop_cfg = load_on_policy_loop_config("configs/on_policy_loop.yaml").model_copy(
                update={
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                }
            )
            data_cfg = load_on_policy_data_config("configs/on_policy_data.yaml")
            loop_queries = [{"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i)} for i in range(6)]

            with patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=data_cfg), \
                patch("post_train.on_policy_loop.load_sft_config"), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.load_query_candidates", return_value=(loop_queries, {"available_rows": 6})), \
                patch(
                    "post_train.on_policy_loop.prepare_on_policy_sft_dataset_from_queries",
                    side_effect=InsufficientRetainedSamplesError("retained 样本不足"),
                ):
                result = run_on_policy_loop(loop_cfg)

        self.assertEqual(result["summary"]["stop_reason"], "insufficient_retained_data")
        self.assertEqual(result["summary"]["completed_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
