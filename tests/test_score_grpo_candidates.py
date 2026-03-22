from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_grpo_candidates import build_progress_status


class ScoreGRPOCandidatesTest(unittest.TestCase):
    def test_build_progress_status_reports_problem_and_completion_counts(self) -> None:
        status = build_progress_status(
            processed_problems=64,
            total_problems=3857,
            num_samples_per_problem=8,
            batch_index=2,
            total_batches=121,
        )
        self.assertIn("problems 64/3857", status)
        self.assertIn("completions 512/30856", status)
        self.assertIn("batch 2/121", status)


if __name__ == "__main__":
    unittest.main()
