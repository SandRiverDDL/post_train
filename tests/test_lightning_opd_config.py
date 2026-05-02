from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_lightning_opd_data_config


class LightningOPDConfigTest(unittest.TestCase):
    def test_default_yaml_loads(self) -> None:
        cfg = load_lightning_opd_data_config("configs/lightning_opd/candidate.yaml")

        self.assertEqual(cfg.sample_size, 1000)
        self.assertEqual(cfg.num_shards, 2)
        self.assertEqual(cfg.top_k, 32)
        self.assertEqual(cfg.gpu0, 0)
        self.assertEqual(cfg.gpu1, 1)

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.yaml"
            path.write_text("sample_size: 1\nunknown_field: true\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_lightning_opd_data_config(path)


if __name__ == "__main__":
    unittest.main()
