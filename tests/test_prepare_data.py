from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_data


class PrepareDataScriptTest(unittest.TestCase):
    def test_main_does_not_fail_before_rng_setup(self) -> None:
        fake_rows = [{"id": "1", "question": "1+1=?", "solution": "Final answer: \\boxed{2}", "final_answer": "2", "problem_type": "Algebra"}]

        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [1, 2, 3]}

        with patch.object(sys, "argv", ["prepare_data.py"]), \
            patch.object(prepare_data.AutoTokenizer, "from_pretrained", return_value=DummyTokenizer()), \
            patch.object(prepare_data, "load_dataset", side_effect=[fake_rows, fake_rows * 200, fake_rows]), \
            patch.object(prepare_data, "sample_numinamath", return_value=fake_rows), \
            patch.object(prepare_data, "write_jsonl"), \
            patch.object(prepare_data, "preview_rows"):
            try:
                prepare_data.main()
            except NameError as exc:  # pragma: no cover
                self.fail(f"prepare_data.main 不应因缺少 random 导入而失败: {exc}")


if __name__ == "__main__":
    unittest.main()
