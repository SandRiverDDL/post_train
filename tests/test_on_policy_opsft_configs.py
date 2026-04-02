from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_on_policy_data_config, load_on_policy_loop_config, load_sft_config


class OnPolicyOpSFTConfigTest(unittest.TestCase):
    def test_on_policy_opsft_sft_config_loads(self) -> None:
        cfg = load_sft_config(ROOT / "configs/on_policy/opsft.yaml")
        self.assertEqual(cfg.loss_mode, "opsft")
        self.assertEqual(cfg.learning_rate, 5.0e-7)
        self.assertEqual(cfg.epochs, 1.0)
        self.assertFalse(cfg.profit_enabled)

    def test_on_policy_opsft_data_config_loads(self) -> None:
        cfg = load_on_policy_data_config(ROOT / "configs/on_policy/data_opsft.yaml")
        self.assertEqual(cfg.temperature, 1.0)

    def test_on_policy_opsft_loop_config_loads(self) -> None:
        cfg = load_on_policy_loop_config(ROOT / "configs/on_policy/loop_opsft.yaml")
        self.assertEqual(cfg.base_on_policy_data_config, Path("configs/on_policy/data_opsft.yaml"))
        self.assertEqual(cfg.base_sft_config, Path("configs/on_policy/opsft.yaml"))
        self.assertEqual(cfg.train_learning_rate, 1.0e-6)
        self.assertEqual(cfg.query_strategy, "candidate_random_mix")
        self.assertEqual(cfg.round_query_count, 640)
        self.assertEqual(cfg.candidate_ratio, 0.5)
        self.assertEqual(cfg.random_ratio, 0.5)
        self.assertEqual(cfg.stop_dataset, Path("data/eval/math500_dev200.jsonl"))
        self.assertEqual(cfg.min_delta, 0.01)
        self.assertTrue(cfg.advance_teacher_on_improvement_only)


if __name__ == "__main__":
    unittest.main()
