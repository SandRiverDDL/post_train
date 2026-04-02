from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.experiments import (
    build_training_run_summary,
    discover_run_summaries,
    register_on_policy_loop_run,
    register_training_run,
    write_experiment_summary_table,
)


class ExperimentsTest(unittest.TestCase):
    def test_register_training_run_collects_dev_and_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "outputs" / "stage1_sft"
            (output_dir / "dev_eval").mkdir(parents=True)
            (tmp / "outputs" / "eval" / "stage1_sft" / "math500").mkdir(parents=True)
            (output_dir / "dev_eval" / "best_checkpoint.json").write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(output_dir / "checkpoint-100"),
                        "global_step": 100,
                        "metrics": {"normalized_accuracy": 0.74},
                        "dev_dataset": "data/eval/math500_dev200.jsonl",
                        "selection_metric": "normalized_accuracy",
                        "ranking_path": str(output_dir / "dev_eval" / "dev_ranking.json"),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (tmp / "outputs" / "eval" / "stage1_sft" / "math500" / "result.json").write_text(
                json.dumps({"metrics": {"pass_at_1": 0.7}}, ensure_ascii=False),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                summary = register_training_run(
                    route="stage1",
                    config_path="configs/stage1/sft.yaml",
                    output_dir=output_dir,
                    train_dataset="data/stage1/train.jsonl",
                    base_model="outputs/stage1_sft_5000/checkpoint-200",
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(summary["dev_metrics"]["normalized_accuracy"], 0.74)
        self.assertEqual(summary["benchmarks"]["math500"]["metrics"]["pass_at_1"], 0.7)
        self.assertEqual(summary["parent_run_id"], "stage1_sft_5000/checkpoint-200")

    def test_register_on_policy_loop_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            loop_root = tmp / "outputs" / "on_policy_loop_opsft_smallpool"
            (tmp / "outputs" / "eval" / "on_policy_loop_opsft_smallpool" / "round3" / "checkpoint-6" / "math500").mkdir(
                parents=True
            )
            (
                tmp / "outputs" / "eval" / "on_policy_loop_opsft_smallpool" / "round3" / "checkpoint-6" / "math500" / "result.json"
            ).write_text(json.dumps({"metrics": {"pass_at_1": 0.702}}, ensure_ascii=False), encoding="utf-8")
            loop_cfg = type(
                "LoopCfg",
                (),
                {
                    "round_base_dir": loop_root,
                    "seed_model": "outputs/stage1_sft_5000/checkpoint-200",
                },
            )()
            final_summary = {
                "best_model_path": "outputs/on_policy_loop_opsft_smallpool/round3/checkpoint-6",
                "best_holdout_accuracy": 0.7467,
                "completed_rounds": 4,
                "best_round_index": 3,
                "stop_reason": "no_improvement",
                "stop_dataset": "data/eval/math500_dev200.jsonl",
            }
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                summary = register_on_policy_loop_run(
                    config_path="configs/on_policy/loop_opsft.yaml",
                    loop_cfg=loop_cfg,
                    final_summary=final_summary,
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(summary["dev_metrics"]["normalized_accuracy"], 0.7467)
        self.assertEqual(summary["benchmarks"]["math500"]["metrics"]["pass_at_1"], 0.702)

    def test_discover_run_summaries_reads_registry_and_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            registry_path = tmp / "experiments" / "registry.jsonl"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "run_id": "stage1_sft",
                        "route": "stage1",
                        "output_dir": "outputs/stage1_sft",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            loop_dir = tmp / "outputs" / "on_policy_loop"
            loop_dir.mkdir(parents=True)
            (loop_dir / "history.json").write_text(
                json.dumps({"seed_model": "outputs/stage1_sft_5000/checkpoint-200", "stop_dataset": "data/eval/math500_dev200.jsonl"}),
                encoding="utf-8",
            )
            (loop_dir / "final_summary.json").write_text(
                json.dumps({"best_model_path": "", "best_holdout_accuracy": 0.73, "completed_rounds": 3, "best_round_index": 2, "stop_reason": "completed"}),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                rows = discover_run_summaries(output_root="outputs", registry_path=registry_path)
            finally:
                os.chdir(old_cwd)

        run_ids = {row["run_id"] for row in rows}
        self.assertIn("stage1_sft", run_ids)
        self.assertIn("outputs/on_policy_loop", run_ids)

    def test_write_experiment_summary_table_outputs_markdown_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rows = [
                {
                    "route": "stage1",
                    "run_id": "outputs/stage1_sft",
                    "source": "registered",
                    "train_dataset": "data/stage1/train.jsonl",
                    "parent_run_id": "outputs/stage1_sft_5000/checkpoint-200",
                    "dev_metrics": {"normalized_accuracy": 0.7267},
                    "benchmarks": {
                        "math500": {"metrics": {"pass_at_1": 0.704}},
                        "gsm8k": {"metrics": {"pass_at_1": 0.846}},
                    },
                },
                {
                    "route": "on_policy",
                    "run_id": "outputs/on_policy_loop",
                    "source": "registered",
                    "train_dataset": "",
                    "parent_run_id": "/home/chy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/abc",
                    "dev_metrics": {"normalized_accuracy": 0.7467},
                    "benchmarks": {
                        "math500": {"metrics": {"pass_at_1": 0.700}},
                        "gsm8k": {"metrics": {"pass_at_1": 0.8476}},
                    },
                },
                {
                    "route": "simpo",
                    "run_id": "outputs/simpo",
                    "source": "discovered",
                    "train_dataset": "",
                    "parent_run_id": "/home/chy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/abc",
                    "dev_metrics": {"normalized_accuracy": 0.4600},
                    "benchmarks": {},
                },
            ]
            md_path, csv_path = write_experiment_summary_table(
                rows=rows,
                markdown_path=tmp / "experiments" / "summary.md",
                csv_path=tmp / "experiments" / "summary.csv",
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(csv_path.exists())
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("Complete Runs", markdown)
            self.assertIn("Incomplete Runs", markdown)
            self.assertIn("Qwen/Qwen2.5-Math-1.5B", markdown)
            self.assertIn("unknown", markdown)
            self.assertNotIn("| Route | Run ID | Train Dataset | Parent | Dev | math500 | gsm8k | Output |", markdown)
            self.assertLess(markdown.find("outputs/stage1_sft"), markdown.find("outputs/on_policy_loop"))


if __name__ == "__main__":
    unittest.main()
