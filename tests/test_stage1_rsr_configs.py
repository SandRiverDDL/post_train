from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config


class Stage1RSRConfigTest(unittest.TestCase):
    def test_stage1_default_sft_config_loads(self) -> None:
        cfg = load_sft_config(ROOT / "configs/stage1/sft.yaml")
        self.assertEqual(cfg.train_dataset, Path("data/stage1_train.jsonl"))
        self.assertEqual(cfg.output_dir, Path("outputs/stage1_sft"))
        self.assertFalse(cfg.profit_enabled)

    def test_stage1_rsr_sft_config_loads(self) -> None:
        cfg = load_sft_config(ROOT / "configs/stage1/rsr_sft.yaml")
        self.assertEqual(cfg.train_dataset, Path("data/stage1_rsr_selected_train.jsonl"))
        self.assertEqual(cfg.output_dir, Path("outputs/stage1_rsr_sft"))
        self.assertFalse(cfg.profit_enabled)
        self.assertEqual(cfg.max_seq_length, 2048)


if __name__ == "__main__":
    unittest.main()
