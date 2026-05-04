from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.eval.reporting import discover_eval_rows, enrich_eval_rows, render_leaderboard


def _write_result(path: Path, *, model: str, dataset: str, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "model": model,
                    "dataset": dataset,
                    "samples": 2,
                    "pass_at_1": score,
                    "pass_at_1_stderr": 0.1,
                    "boxed_rate": 1.0,
                    "avg_output_tokens": 12.0,
                }
            }
        ),
        encoding="utf-8",
    )


class EvalReportingTest(unittest.TestCase):
    def test_discover_eval_rows_prefers_result_json_over_manual_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model = "outputs/demo/checkpoint-1"
            dataset = "data/eval/math500_test.jsonl"
            _write_result(tmp / "eval" / "demo" / "math500" / "result.json", model=model, dataset=dataset, score=0.7)
            manual = tmp / "results.md"
            manual.write_text(
                json.dumps(
                    {
                        "model": model,
                        "dataset": dataset,
                        "metrics": {"pass_at_1": 0.1, "samples": 2},
                    }
                ),
                encoding="utf-8",
            )

            rows = discover_eval_rows(eval_root=tmp / "eval", manual_result_files=[manual])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "result_json")
        self.assertEqual(rows[0]["pass_at_1"], 0.7)

    def test_render_leaderboard_keeps_dev_only_rows_out_of_main_table(self) -> None:
        rows = [
            {
                "source": "result_json",
                "result_path": "outputs/eval/dev/result.json",
                "model": "outputs/grpo/checkpoint-1",
                "dataset": "data/eval/math500_dev200.jsonl",
                "task": "math500_dev200",
                "samples": 200,
                "pass_at_1": 0.9,
                "stderr": 0.1,
                "boxed_rate": 1.0,
                "avg_output_tokens": 10.0,
            },
            {
                "source": "result_json",
                "result_path": "outputs/eval/main/result.json",
                "model": "outputs/sft/checkpoint-1",
                "dataset": "data/eval/math500_test.jsonl",
                "task": "math500",
                "samples": 500,
                "pass_at_1": 0.6,
                "stderr": 0.1,
                "boxed_rate": 1.0,
                "avg_output_tokens": 10.0,
            },
        ]
        enriched = enrich_eval_rows(
            rows,
            {
                "models": {
                    "outputs/grpo/checkpoint-1": {"method": "GRPO", "label": "dev only"},
                    "outputs/sft/checkpoint-1": {"method": "SFT", "label": "tracked sft"},
                }
            },
        )

        markdown = render_leaderboard(enriched)
        main_table = markdown.split("## Task Details", maxsplit=1)[0]

        self.assertIn("tracked sft", main_table)
        self.assertNotIn("dev only", main_table)
        self.assertIn("dev only", markdown)

    def test_result_path_task_suffix_test_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model = "outputs/demo/checkpoint-1"
            dataset = "data/eval/math500_test.jsonl"
            _write_result(tmp / "eval" / "demo" / "math500_test" / "result.json", model=model, dataset=dataset, score=0.7)

            rows = discover_eval_rows(eval_root=tmp / "eval", manual_result_files=[])

        self.assertEqual(rows[0]["task"], "math500")

    def test_untracked_models_are_hidden_from_leaderboard(self) -> None:
        rows = [
            {
                "source": "result_json",
                "result_path": "outputs/eval/noisy/result.json",
                "model": "outputs/noisy/checkpoint-999",
                "dataset": "data/eval/math500_test.jsonl",
                "task": "math500",
                "samples": 500,
                "pass_at_1": 0.99,
            }
        ]

        enriched = enrich_eval_rows(rows, {"models": {}})
        markdown = render_leaderboard(enriched)

        self.assertNotIn("noisy/checkpoint-999", markdown)
        self.assertIn("当前隐藏 1 行", markdown)

    def test_metadata_can_ignore_bad_runs(self) -> None:
        rows = [
            {
                "source": "result_json",
                "result_path": "outputs/eval/bad/result.json",
                "model": "outputs/bad/checkpoint-1",
                "dataset": "data/eval/math500_test.jsonl",
                "task": "math500",
                "samples": 500,
                "pass_at_1": 0.1,
            }
        ]

        enriched = enrich_eval_rows(
            rows,
            {"models": {"outputs/bad/checkpoint-1": {"method": "Bad", "label": "bad run", "ignore": True}}},
        )
        markdown = render_leaderboard(enriched)

        self.assertNotIn("| Bad | bad run |", markdown.split("## Task Details", maxsplit=1)[0])
        self.assertIn("## Ignored / Archived", markdown)
        self.assertIn("bad run", markdown)


if __name__ == "__main__":
    unittest.main()
