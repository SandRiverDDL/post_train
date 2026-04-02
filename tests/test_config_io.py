from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config.io import load_yaml_config


class ConfigIoTest(unittest.TestCase):
    def test_load_yaml_config_supports_extends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "base.yaml").write_text(
                "\n".join(
                    [
                        "model_name: base-model",
                        "batch_size: 4",
                        "nested:",
                        "  lr: 1.0e-4",
                        "  tags:",
                        "    - a",
                        "    - b",
                    ]
                ),
                encoding="utf-8",
            )
            (tmp_path / "child.yaml").write_text(
                "\n".join(
                    [
                        "extends: base.yaml",
                        "batch_size: 8",
                        "nested:",
                        "  lr: 5.0e-5",
                        "  tags:",
                        "    - c",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_yaml_config(tmp_path / "child.yaml")

            self.assertEqual(cfg["model_name"], "base-model")
            self.assertEqual(cfg["batch_size"], 8)
            self.assertEqual(cfg["nested"]["lr"], 5.0e-5)
            self.assertEqual(cfg["nested"]["tags"], ["c"])

    def test_load_yaml_config_detects_extend_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "循环引用"):
                load_yaml_config(tmp_path / "a.yaml")


if __name__ == "__main__":
    unittest.main()
