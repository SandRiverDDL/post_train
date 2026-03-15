from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from score_grpo_candidates import (
    build_cache_key,
    build_sampling_spec,
    resolve_prompt,
    summarize_trial_results,
)
from select_grpo_scored_subset import _largest_remainder_sample, passes_difficulty_filter


class RejectionSamplingTest(unittest.TestCase):
    def test_resolve_prompt_prefers_existing_prompt(self) -> None:
        row = {
            "question": "1+1=?",
            "prompt": "custom prompt",
        }
        self.assertEqual(resolve_prompt(row), "custom prompt")

    def test_build_cache_key_changes_with_sampling(self) -> None:
        row = {"id": "1", "question": "1+1=?", "final_answer": "2"}

        class Args:
            num_samples_per_problem = 4
            temperature = 0.8
            top_p = 0.95
            max_new_tokens = 256

        sampling = build_sampling_spec(Args())
        key_a = build_cache_key(row, prompt="prompt", model_id="m1", sampling=sampling)
        sampling["temperature"] = 0.9
        key_b = build_cache_key(row, prompt="prompt", model_id="m1", sampling=sampling)
        self.assertNotEqual(key_a, key_b)

    def test_summarize_trial_results_aggregates_metrics(self) -> None:
        result = summarize_trial_results(
            [
                "Reasoning\n\nFinal answer: \\boxed{4}",
                "Reasoning only",
                "Wrong\n\nFinal answer: \\boxed{5}",
                "Final answer: \\boxed{4}",
            ],
            [12, 8, 10, 6],
            "4",
        )
        self.assertEqual(result["sft_num_trials"], 4)
        self.assertEqual(result["sft_correct_count"], 2)
        self.assertEqual(result["sft_parse_count"], 3)
        self.assertEqual(result["sft_format_count"], 3)
        self.assertAlmostEqual(result["sft_correct_rate"], 0.5)
        self.assertAlmostEqual(result["sft_parse_rate"], 0.75)
        self.assertAlmostEqual(result["sft_mean_generated_tokens"], 9.0)
        self.assertEqual(len(result["preview_samples"]), 2)

    def test_passes_difficulty_filter_honors_bounds(self) -> None:
        row = {"sft_correct_rate": 0.25, "sft_parse_rate": 0.5}
        self.assertTrue(
            passes_difficulty_filter(
                row,
                min_correct_rate=0.25,
                max_correct_rate=0.5,
                min_parse_rate=0.5,
            )
        )
        self.assertFalse(
            passes_difficulty_filter(
                {"sft_correct_rate": 0.75, "sft_parse_rate": 1.0},
                min_correct_rate=0.25,
                max_correct_rate=0.5,
                min_parse_rate=0.5,
            )
        )

    def test_largest_remainder_sample_keeps_target_size(self) -> None:
        rows = [
            {"id": "1", "problem_type": "A"},
            {"id": "2", "problem_type": "A"},
            {"id": "3", "problem_type": "A"},
            {"id": "4", "problem_type": "B"},
        ]
        sampled = _largest_remainder_sample(rows, target_size=2, stratify_by="problem_type", seed=42)
        self.assertEqual(len(sampled), 2)
        self.assertEqual({row["problem_type"] for row in sampled}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
