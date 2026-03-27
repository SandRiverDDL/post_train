from __future__ import annotations

import json
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

        self.assertEqual(cfg.seed_model, "outputs/stage1_sft_5000/checkpoint-200")
        self.assertEqual(cfg.max_rounds, 10)
        self.assertEqual(cfg.round_query_count, 256)
        self.assertEqual(cfg.patience, 3)
        self.assertEqual(cfg.min_delta, 0.0)

    def test_build_round_paths_uses_round_index(self) -> None:
        cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml")
        paths = build_round_paths(cfg, 2)

        self.assertEqual(paths["data_dir"], Path("data/on_policy_loop/round2"))
        self.assertEqual(paths["output_dir"], Path("outputs/on_policy_loop/round2"))

    def test_build_round_configs_override_round_specific_fields(self) -> None:
        loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml")
        data_cfg = build_round_data_config(
            load_on_policy_data_config("configs/on_policy/data.yaml"),
            loop_cfg,
            round_index=3,
            generation_model="model-x",
        )
        base_train_cfg = load_sft_config("configs/on_policy/sft.yaml")
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
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "query_strategy": "uniform_epoch",
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                    "max_rounds": 5,
                    "patience": 3,
                    "round_query_count": 5,
                    "seed_model": "seed-model",
                }
            )
            data_result = {
                "retained_output_path": str(tmp_path / "data/round/train.jsonl"),
                "report_path": str(tmp_path / "data/round/train.report.json"),
                "report": {"retained": {"kept": 100, "retained_ratio": 0.5}},
            }
            holdout_scores = [0.70, 0.70, 0.69, 0.69]
            data_cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
            loop_queries = [
                {"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i), "meta": {"source": "toy"}}
                for i in range(12)
            ]

            def fake_holdout(**kwargs):
                score = holdout_scores.pop(0)
                task_root = Path(kwargs["output_dir"]) / "global_dev_math500_150"
                return {
                    "result": {"metrics": {"normalized_accuracy": score}},
                    "result_path": str(task_root / "result.json"),
                    "raw_result_path": str(task_root / "raw.json"),
                }

            def fake_sample_round_queries(query_rows, sampler_state, *, requested_count):
                self.assertEqual(requested_count, 5)
                return loop_queries[:requested_count], {
                    "query_pool_size": len(loop_queries),
                    "requested_query_count": requested_count,
                    "effective_query_count": requested_count,
                    "crossed_epoch": False,
                    "epoch_index_start": 0,
                    "epoch_offset_start": 0,
                    "epoch_index_end": 0,
                    "epoch_offset_end": requested_count,
                }

            with patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=data_cfg), \
                patch("post_train.on_policy_loop.load_sft_config", return_value=load_sft_config("configs/on_policy/sft.yaml")), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.load_query_candidates", return_value=(loop_queries, {"available_rows": 12})), \
                patch("post_train.on_policy_loop.sample_round_queries", side_effect=fake_sample_round_queries), \
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
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "query_strategy": "uniform_epoch",
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                }
            )
            data_cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
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

    def test_run_on_policy_loop_records_mixed_strategy_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "query_strategy": "mixed_bootstrap_candidate",
                    "bootstrap_all_correct_raw_samples": tmp_path / "bootstrap.jsonl",
                    "candidate_query_file": tmp_path / "candidate.jsonl",
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                    "max_rounds": 1,
                    "round_query_count": 5,
                    "seed_model": "seed-model",
                }
            )
            data_result = {
                "raw_samples_output_path": str(tmp_path / "data/round/raw_samples.jsonl"),
                "retained_output_path": str(tmp_path / "data/round/train.jsonl"),
                "report_path": str(tmp_path / "data/round/train.report.json"),
                "report": {"retained": {"kept": 5, "retained_ratio": 1.0}},
            }
            mixed_round_rows = [
                {"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i), "meta": {"source": "toy"}}
                for i in range(5)
            ]

            with patch(
                "post_train.on_policy_loop.build_mixed_query_strategy",
                return_value={
                    "state_path": tmp_path / "data/query_strategy/state.json",
                    "bootstrap_rows": [{"id": "b1"}],
                    "candidate_rows": [{"id": "c1"}],
                },
            ), \
                patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=load_on_policy_data_config("configs/on_policy/data.yaml")), \
                patch("post_train.on_policy_loop.load_sft_config", return_value=load_sft_config("configs/on_policy/sft.yaml")), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.sample_mixed_round_queries", return_value=(mixed_round_rows, {"effective_query_count": 5, "source_mix": {"candidate_frozen_count": 1}})), \
                patch("post_train.on_policy_loop.prepare_on_policy_sft_dataset_from_queries", return_value=data_result), \
                patch("post_train.on_policy_loop.update_mixed_query_strategy", return_value={"candidate_newly_frozen": 1, "candidate_frozen_count": 2}), \
                patch("post_train.on_policy_loop.train_sft") as mock_train, \
                patch("post_train.on_policy_loop.evaluate_single_dataset", return_value={"result": {"metrics": {"normalized_accuracy": 0.8}}, "result_path": str(tmp_path / "result.json"), "raw_result_path": str(tmp_path / "raw.json")}):
                mock_train.side_effect = lambda cfg: Path(cfg.output_dir)
                result = run_on_policy_loop(loop_cfg)

        self.assertEqual(result["summary"]["query_strategy"], "mixed_bootstrap_candidate")
        self.assertEqual(result["summary"]["query_strategy_state_path"], str(tmp_path / "data/query_strategy/state.json"))

    def test_run_on_policy_loop_rejects_existing_dirs_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "round_base_dir": tmp_path / "outputs",
                    "data_base_dir": tmp_path / "data",
                }
            )
            loop_cfg.round_base_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "--resume"):
                run_on_policy_loop(loop_cfg)

    def test_run_on_policy_loop_resume_continues_from_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            round_base_dir = tmp_path / "outputs"
            data_base_dir = tmp_path / "data"
            round_base_dir.mkdir(parents=True)
            data_base_dir.mkdir(parents=True)
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "query_strategy": "uniform_epoch",
                    "round_base_dir": round_base_dir,
                    "data_base_dir": data_base_dir,
                    "max_rounds": 2,
                    "round_query_count": 5,
                    "seed_model": "seed-model",
                }
            )
            history = {
                "seed_model": "seed-model",
                "stop_dataset": "data/eval/global_dev_math500_150.jsonl",
                "patience": 3,
                "min_delta": 0.0,
                "max_rounds": 2,
                "query_strategy": "uniform_epoch",
                "best_round_index": 1,
                "best_holdout_accuracy": 0.7,
                "best_model_path": str(round_base_dir / "round1"),
                "current_model": str(round_base_dir / "round1"),
                "no_improve_rounds": 0,
                "last_completed_round": 1,
                "stop_reason": "",
                "query_sampler_state": {
                    "seed": 42,
                    "pool_size": 12,
                    "epoch_index": 0,
                    "epoch_offset": 5,
                },
                "query_strategy_state_path": "",
                "rounds": [
                    {
                        "round_index": 1,
                        "round_name": "round1",
                        "round_model_path": str(round_base_dir / "round1"),
                    }
                ],
            }
            (round_base_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
            data_result = {
                "retained_output_path": str(data_base_dir / "round2/train.jsonl"),
                "report_path": str(data_base_dir / "round2/train.report.json"),
                "report": {"retained": {"kept": 8, "retained_ratio": 0.8}},
            }
            loop_queries = [
                {"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i), "meta": {"source": "toy"}}
                for i in range(12)
            ]

            def fake_sample_round_queries(query_rows, sampler_state, *, requested_count):
                self.assertEqual(int(sampler_state["epoch_offset"]), 5)
                return loop_queries[5:10], {
                    "query_pool_size": len(loop_queries),
                    "requested_query_count": requested_count,
                    "effective_query_count": requested_count,
                    "crossed_epoch": False,
                    "epoch_index_start": 0,
                    "epoch_offset_start": 5,
                    "epoch_index_end": 0,
                    "epoch_offset_end": 10,
                }

            with patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=load_on_policy_data_config("configs/on_policy/data.yaml")), \
                patch("post_train.on_policy_loop.load_sft_config", return_value=load_sft_config("configs/on_policy/sft.yaml")), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.load_query_candidates", return_value=(loop_queries, {"available_rows": 12})), \
                patch("post_train.on_policy_loop.sample_round_queries", side_effect=fake_sample_round_queries), \
                patch("post_train.on_policy_loop.prepare_on_policy_sft_dataset_from_queries", return_value=data_result), \
                patch("post_train.on_policy_loop.train_sft") as mock_train, \
                patch("post_train.on_policy_loop.evaluate_single_dataset", return_value={"result": {"metrics": {"normalized_accuracy": 0.8}}, "result_path": str(tmp_path / "result.json"), "raw_result_path": str(tmp_path / "raw.json")}):
                mock_train.side_effect = lambda cfg: Path(cfg.output_dir)
                result = run_on_policy_loop(loop_cfg, resume=True)

        self.assertEqual(result["summary"]["completed_rounds"], 2)
        self.assertEqual(result["summary"]["best_round_index"], 2)
        self.assertEqual(result["summary"]["best_model_path"], str(round_base_dir / "round2"))

    def test_run_on_policy_loop_rejects_resume_for_finished_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            round_base_dir = tmp_path / "outputs"
            data_base_dir = tmp_path / "data"
            round_base_dir.mkdir(parents=True)
            data_base_dir.mkdir(parents=True)
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "round_base_dir": round_base_dir,
                    "data_base_dir": data_base_dir,
                }
            )
            history = {
                "seed_model": "seed-model",
                "stop_reason": "no_improvement",
                "rounds": [],
            }
            (round_base_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "自然结束"):
                run_on_policy_loop(loop_cfg, resume=True)

    def test_run_on_policy_loop_overwrite_removes_stale_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            round_base_dir = tmp_path / "outputs"
            data_base_dir = tmp_path / "data"
            (round_base_dir / "stale.txt").parent.mkdir(parents=True)
            (round_base_dir / "stale.txt").write_text("old", encoding="utf-8")
            (data_base_dir / "stale.txt").parent.mkdir(parents=True)
            (data_base_dir / "stale.txt").write_text("old", encoding="utf-8")
            loop_cfg = load_on_policy_loop_config("configs/on_policy/loop.yaml").model_copy(
                update={
                    "query_strategy": "uniform_epoch",
                    "round_base_dir": round_base_dir,
                    "data_base_dir": data_base_dir,
                }
            )
            data_cfg = load_on_policy_data_config("configs/on_policy/data.yaml")
            loop_queries = [{"id": f"q{i}", "question": f"Question {i}", "final_answer": str(i)} for i in range(6)]

            with patch("post_train.on_policy_loop.load_on_policy_data_config", return_value=data_cfg), \
                patch("post_train.on_policy_loop.load_sft_config"), \
                patch("post_train.on_policy_loop.load_eval_config"), \
                patch("post_train.on_policy_loop.load_query_candidates", return_value=(loop_queries, {"available_rows": 6})), \
                patch(
                    "post_train.on_policy_loop.prepare_on_policy_sft_dataset_from_queries",
                    side_effect=InsufficientRetainedSamplesError("retained 样本不足"),
                ):
                result = run_on_policy_loop(loop_cfg, overwrite=True)

        self.assertFalse((round_base_dir / "stale.txt").exists())
        self.assertFalse((data_base_dir / "stale.txt").exists())
        self.assertEqual(result["summary"]["stop_reason"], "insufficient_retained_data")


if __name__ == "__main__":
    unittest.main()
