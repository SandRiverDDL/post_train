from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grpo_checkpoints import collect_model_dirs, parse_args


class EvalGRPOCheckpointsTest(unittest.TestCase):
    def test_collect_model_dirs_sorts_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoint-50").mkdir()
            (run_dir / "checkpoint-10").mkdir()
            (run_dir / "checkpoint-100").mkdir()

            model_dirs = collect_model_dirs(run_dir, include_final=False)
            self.assertEqual([path.name for path in model_dirs], ["checkpoint-10", "checkpoint-50", "checkpoint-100"])

    def test_collect_model_dirs_can_include_final_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoint-25").mkdir()
            (run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            model_dirs = collect_model_dirs(run_dir, include_final=True)
            self.assertEqual([path.name for path in model_dirs], ["checkpoint-25", run_dir.name])

    def test_parse_args_defaults_to_eval_config(self) -> None:
        with patch.object(sys, "argv", ["eval_grpo_checkpoints.py", "--run-dir", "outputs/grpo"]):
            args = parse_args()
        self.assertEqual(args.config, "configs/eval.yaml")


if __name__ == "__main__":
    unittest.main()
