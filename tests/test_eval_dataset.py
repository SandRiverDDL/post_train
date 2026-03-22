from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_dataset import parse_args


class EvalDatasetScriptTest(unittest.TestCase):
    def test_parse_args_defaults_to_eval_config(self) -> None:
        with patch.object(sys, "argv", ["eval_dataset.py"]):
            args = parse_args()
        self.assertEqual(args.config, "configs/eval.yaml")

    def test_parse_args_defaults_to_official_mode(self) -> None:
        with patch.object(sys, "argv", ["eval_dataset.py"]):
            args = parse_args()
        self.assertEqual(args.mode, "official")


if __name__ == "__main__":
    unittest.main()
