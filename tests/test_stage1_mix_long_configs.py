from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config, load_stage2_mix_long_data_config


class Stage1MixLongConfigTest(unittest.TestCase):
    def test_stage1_mix_long_data_config_loads(self) -> None:
        cfg = load_stage2_mix_long_data_config(ROOT / "configs/stage1/mix_long_data.yaml")
        self.assertEqual(cfg.dataset_name, "UWNSL/Mix-Long_long_0.2_short_0.8")
        self.assertEqual(cfg.max_solution_tokens, 4096)
        self.assertIsNone(cfg.sample_size)
        self.assertEqual(cfg.output_path, Path("data/stage1/mix_long/train.jsonl"))

    def test_stage1_mix_long_sft_config_loads(self) -> None:
        cfg = load_sft_config(ROOT / "configs/stage1/mix_long_sft.yaml")
        self.assertEqual(cfg.train_dataset, Path("data/stage1/mix_long/train.jsonl"))
        self.assertEqual(cfg.output_dir, Path("outputs/stage1_mix_long_sft"))
        self.assertEqual(cfg.max_seq_length, 5120)
        self.assertEqual(cfg.epochs, 2.0)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.gradient_accumulation_steps, 8)


if __name__ == "__main__":
    unittest.main()
