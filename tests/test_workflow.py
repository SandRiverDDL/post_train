from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import EvalConfig, EvalTaskConfig, GRPOTrainConfig, WorkflowConfig, load_workflow_config
from post_train.workflow import run_workflow


class WorkflowTest(unittest.TestCase):
    def test_load_workflow_config_supports_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workflow.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "route: grpo",
                        "train_config: configs/grpo/train_cheap.yaml",
                        "eval_config: configs/eval/default.yaml",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_workflow_config(config_path)

        self.assertEqual(cfg.route, "grpo")
        self.assertEqual(cfg.benchmark_tasks, ["math500", "gsm8k"])
        self.assertEqual(cfg.baseline_policy, "reuse_or_eval")
        self.assertEqual(cfg.failure_policy, "stop")

    def test_run_workflow_reuses_baseline_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "outputs" / "grpo_run"
            eval_root = tmp / "outputs" / "eval"
            baseline_root = eval_root / "external" / "base" / "model"
            (baseline_root / "math500").mkdir(parents=True)
            (baseline_root / "gsm8k").mkdir(parents=True)
            (baseline_root / "math500" / "result.json").write_text(
                json.dumps({"metrics": {"pass_at_1": 0.50}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (baseline_root / "gsm8k" / "result.json").write_text(
                json.dumps({"metrics": {"pass_at_1": 0.60}}, ensure_ascii=False),
                encoding="utf-8",
            )
            workflow_cfg = WorkflowConfig(
                route="grpo",
                train_config=Path("configs/grpo/train_cheap.yaml"),
                eval_config=Path("configs/eval/default.yaml"),
                dev_dataset=Path("data/eval/math500_dev200.jsonl"),
            )
            train_cfg = GRPOTrainConfig(
                base_model_name="Qwen/Qwen2.5-Math-1.5B",
                model_name_or_path="base/model",
                train_dataset=Path("data/grpo/train.jsonl"),
                output_dir=output_dir,
            )
            eval_cfg = EvalConfig(
                model_name="Qwen/Qwen2.5-Math-1.5B",
                output_dir=eval_root,
                tasks=[
                    EvalTaskConfig(name="math500", dataset_path=Path("data/eval/math500_test.jsonl")),
                    EvalTaskConfig(name="gsm8k", dataset_path=Path("data/eval/gsm8k_test.jsonl")),
                ],
            )
            eval_calls: list[tuple[str, str]] = []

            def fake_run_eval_task(*, model_name, task, eval_cfg, batch_size, max_new_tokens, limit, output_dir=None, max_lora_rank=None):
                eval_calls.append((str(model_name), task.name))
                metrics_map = {
                    "math500": {"pass_at_1": 0.70},
                    "gsm8k": {"pass_at_1": 0.80},
                }
                result_dir = Path(output_dir or eval_cfg.output_dir) / Path(model_name).name / task.name
                result_dir.mkdir(parents=True, exist_ok=True)
                result_payload = {"metrics": metrics_map[task.name]}
                result_path = result_dir / "result.json"
                result_path.write_text(json.dumps(result_payload, ensure_ascii=False), encoding="utf-8")
                raw_path = result_dir / "raw.json"
                raw_path.write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")
                return {
                    "result": result_payload,
                    "result_path": str(result_path),
                    "raw_result_path": str(raw_path),
                    "raw_result": {"ok": True},
                }

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                with (
                    patch("post_train.workflow.load_grpo_train_config", return_value=train_cfg),
                    patch("post_train.workflow.load_eval_config", return_value=eval_cfg),
                    patch("post_train.workflow.load_grpo_reward_config", return_value=object()),
                    patch("post_train.workflow.train_grpo", return_value=output_dir),
                    patch(
                        "post_train.workflow.run_checkpoint_selection",
                        return_value={
                            "best": {
                                "checkpoint_path": str(output_dir / "checkpoint-25"),
                                "global_step": 25,
                                "metrics": {"normalized_accuracy": 0.76},
                                "result_path": str(output_dir / "dev_eval" / "checkpoint-25" / "result.json"),
                                "raw_result_path": str(output_dir / "dev_eval" / "checkpoint-25" / "raw.json"),
                            },
                            "ranking_path": str(output_dir / "dev_eval" / "dev_ranking.json"),
                            "best_path": str(output_dir / "dev_eval" / "best_checkpoint.json"),
                        },
                    ),
                    patch("post_train.workflow.run_eval_task", side_effect=fake_run_eval_task),
                    patch(
                        "post_train.workflow.register_training_run",
                        return_value={"summary_path": "outputs/grpo_run/run_summary.json"},
                    ),
                ):
                    result = run_workflow(workflow_cfg)
                summary_path = output_dir / "workflow_summary.json"
                self.assertTrue(summary_path.exists())
            finally:
                os.chdir(old_cwd)

        self.assertEqual(
            eval_calls,
            [
                (str(output_dir / "checkpoint-25"), "math500"),
                (str(output_dir / "checkpoint-25"), "gsm8k"),
            ],
        )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["comparison"]["overall_status"], "improved")
        self.assertAlmostEqual(result["comparison"]["tasks"]["math500"]["delta"], 0.20)
        self.assertAlmostEqual(result["comparison"]["tasks"]["gsm8k"]["delta"], 0.20)

    def test_run_workflow_falls_back_to_baseline_eval_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "outputs" / "grpo_run"
            workflow_cfg = WorkflowConfig(
                route="grpo",
                train_config=Path("configs/grpo/train_cheap.yaml"),
                eval_config=Path("configs/eval/default.yaml"),
                dev_dataset=Path("data/eval/math500_dev200.jsonl"),
            )
            train_cfg = GRPOTrainConfig(
                base_model_name="Qwen/Qwen2.5-Math-1.5B",
                model_name_or_path="base/model",
                train_dataset=Path("data/grpo/train.jsonl"),
                output_dir=output_dir,
            )
            eval_cfg = EvalConfig(
                model_name="Qwen/Qwen2.5-Math-1.5B",
                output_dir=tmp / "outputs" / "eval",
                tasks=[
                    EvalTaskConfig(name="math500", dataset_path=Path("data/eval/math500_test.jsonl")),
                    EvalTaskConfig(name="gsm8k", dataset_path=Path("data/eval/gsm8k_test.jsonl")),
                ],
            )
            eval_calls: list[tuple[str, str]] = []

            def fake_run_eval_task(*, model_name, task, eval_cfg, batch_size, max_new_tokens, limit, output_dir=None, max_lora_rank=None):
                eval_calls.append((str(model_name), task.name))
                score = 0.70 if "checkpoint-25" in str(model_name) else 0.65
                result_payload = {"metrics": {"pass_at_1": score}}
                result_dir = Path(output_dir or eval_cfg.output_dir) / Path(model_name).name / task.name
                result_dir.mkdir(parents=True, exist_ok=True)
                result_path = result_dir / "result.json"
                result_path.write_text(json.dumps(result_payload, ensure_ascii=False), encoding="utf-8")
                raw_path = result_dir / "raw.json"
                raw_path.write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")
                return {
                    "result": result_payload,
                    "result_path": str(result_path),
                    "raw_result_path": str(raw_path),
                    "raw_result": {"ok": True},
                }

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                with (
                    patch("post_train.workflow.load_grpo_train_config", return_value=train_cfg),
                    patch("post_train.workflow.load_eval_config", return_value=eval_cfg),
                    patch("post_train.workflow.load_grpo_reward_config", return_value=object()),
                    patch("post_train.workflow.train_grpo", return_value=output_dir),
                    patch(
                        "post_train.workflow.run_checkpoint_selection",
                        return_value={
                            "best": {
                                "checkpoint_path": str(output_dir / "checkpoint-25"),
                                "global_step": 25,
                                "metrics": {"normalized_accuracy": 0.76},
                                "result_path": str(output_dir / "dev_eval" / "checkpoint-25" / "result.json"),
                                "raw_result_path": str(output_dir / "dev_eval" / "checkpoint-25" / "raw.json"),
                            },
                            "ranking_path": str(output_dir / "dev_eval" / "dev_ranking.json"),
                            "best_path": str(output_dir / "dev_eval" / "best_checkpoint.json"),
                        },
                    ),
                    patch("post_train.workflow.run_eval_task", side_effect=fake_run_eval_task),
                    patch(
                        "post_train.workflow.register_training_run",
                        return_value={"summary_path": "outputs/grpo_run/run_summary.json"},
                    ),
                ):
                    result = run_workflow(workflow_cfg)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(
            eval_calls,
            [
                (str(output_dir / "checkpoint-25"), "math500"),
                (str(output_dir / "checkpoint-25"), "gsm8k"),
                ("base/model", "math500"),
                ("base/model", "gsm8k"),
            ],
        )
        self.assertEqual(result["comparison"]["overall_status"], "improved")

    def test_run_workflow_stops_and_writes_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "outputs" / "grpo_run"
            workflow_cfg = WorkflowConfig(
                route="grpo",
                train_config=Path("configs/grpo/train_cheap.yaml"),
                eval_config=Path("configs/eval/default.yaml"),
                dev_dataset=Path("data/eval/math500_dev200.jsonl"),
            )
            train_cfg = GRPOTrainConfig(
                base_model_name="Qwen/Qwen2.5-Math-1.5B",
                model_name_or_path="base/model",
                train_dataset=Path("data/grpo/train.jsonl"),
                output_dir=output_dir,
            )
            eval_cfg = EvalConfig(
                model_name="Qwen/Qwen2.5-Math-1.5B",
                dataset_path=Path("data/eval/math500_dev200.jsonl"),
            )

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                with (
                    patch("post_train.workflow.load_grpo_train_config", return_value=train_cfg),
                    patch("post_train.workflow.load_eval_config", return_value=eval_cfg),
                    patch("post_train.workflow.load_grpo_reward_config", return_value=object()),
                    patch("post_train.workflow.train_grpo", side_effect=RuntimeError("boom")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        run_workflow(workflow_cfg)
                failure_path = output_dir / "workflow_failure.json"
                self.assertTrue(failure_path.exists())
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["failure_stage"], "training")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
