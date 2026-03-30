from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config
from post_train.sft_selection import checkpoint_step, scan_checkpoint_dirs, select_best_checkpoint


class SFTSelectionTest(unittest.TestCase):
    def test_load_sft_config_supports_checkpoint_save_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "stage1.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1_train.jsonl",
                        "output_dir: outputs/stage1_sft",
                        "epochs: 2",
                        "loss_mode: opsft",
                        "profit_enabled: true",
                        "profit_threshold: 0.2",
                        "save_strategy: steps",
                        "save_steps: 25",
                        "save_total_limit: 12",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "opsft 与 profit_enabled 不能同时开启"):
                load_sft_config(config_path)

    def test_load_sft_config_supports_opsft_loss_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "opsft.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1_train.jsonl",
                        "output_dir: outputs/stage1_sft",
                        "epochs: 1",
                        "loss_mode: opsft",
                        "profit_enabled: false",
                        'save_strategy: "no"',
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.epochs, 1)
        self.assertEqual(cfg.loss_mode, "opsft")
        self.assertFalse(cfg.profit_enabled)

    def test_scan_checkpoint_dirs_sorts_by_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "checkpoint-100").mkdir()
            (output_dir / "checkpoint-25").mkdir()
            (output_dir / "checkpoint-50").mkdir()
            (output_dir / "logs").mkdir()

            checkpoints = scan_checkpoint_dirs(output_dir)

        self.assertEqual([path.name for path in checkpoints], ["checkpoint-25", "checkpoint-50", "checkpoint-100"])
        self.assertEqual(checkpoint_step(checkpoints[-1]), 100)

    def test_select_best_checkpoint_prefers_accuracy_then_format(self) -> None:
        records = [
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-25",
                "global_step": 25,
                "metrics": {
                    "normalized_accuracy": 0.80,
                    "boxed_rate": 0.95,
                    "parse_success_rate": 0.95,
                },
            },
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-50",
                "global_step": 50,
                "metrics": {
                    "normalized_accuracy": 0.80,
                    "boxed_rate": 0.98,
                    "parse_success_rate": 0.97,
                },
            },
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-75",
                "global_step": 75,
                "metrics": {
                    "normalized_accuracy": 0.78,
                    "boxed_rate": 1.0,
                    "parse_success_rate": 1.0,
                },
            },
        ]

        best = select_best_checkpoint(records)

        self.assertEqual(best["checkpoint_path"], "outputs/stage1_sft/checkpoint-50")


if __name__ == "__main__":
    unittest.main()
