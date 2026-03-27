from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.aime_eval import (
    prepare_aime_eval_artifact,
    resolve_aime_dataset,
)


class AimeEvalTest(unittest.TestCase):
    def test_resolve_aime_dataset_returns_expected_defaults(self) -> None:
        aime24 = resolve_aime_dataset(24)
        aime25 = resolve_aime_dataset(25)

        self.assertEqual(aime24["dataset_name"], "math-ai/aime24")
        self.assertEqual(aime24["source"], "aime24")
        self.assertEqual(aime24["output"], "data/eval/aime24_test.jsonl")
        self.assertEqual(aime25["dataset_name"], "math-ai/aime25")
        self.assertEqual(aime25["source"], "aime25")
        self.assertEqual(aime25["output"], "data/eval/aime25_test.jsonl")

    def test_prepare_aime_eval_artifact_supports_aime24_and_aime25(self) -> None:
        raw_rows = [{"id": "a-1", "problem": "Compute 1+1.", "answer": "2"}]
        with patch("post_train.aime_eval.prepare_benchmark_artifact", return_value=[{"id": "a-1"}]) as mock_prepare:
            rows24 = prepare_aime_eval_artifact(year=24)
            rows25 = prepare_aime_eval_artifact(year=25)

        self.assertEqual(rows24, [{"id": "a-1"}])
        self.assertEqual(rows25, [{"id": "a-1"}])
        self.assertEqual(mock_prepare.call_args_list[0].kwargs["dataset_name"], "math-ai/aime24")
        self.assertEqual(mock_prepare.call_args_list[0].kwargs["source"], "aime24")
        self.assertEqual(mock_prepare.call_args_list[1].kwargs["dataset_name"], "math-ai/aime25")
        self.assertEqual(mock_prepare.call_args_list[1].kwargs["source"], "aime25")


if __name__ == "__main__":
    unittest.main()
