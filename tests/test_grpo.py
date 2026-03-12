from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import GRPOTrainConfig
from rl.grpo import (
    build_grpo_prompt,
    build_grpo_record,
    correctness_reward,
    default_reward_weights,
    format_reward,
    parse_reward,
)


class GRPOUtilsTest(unittest.TestCase):
    def test_build_grpo_prompt_uses_protocol_contract(self) -> None:
        prompt = build_grpo_prompt("Compute 2+2.")
        self.assertIn("Question:\nCompute 2+2.", prompt)
        self.assertIn("Final answer: \\boxed{...}", prompt)

    def test_build_grpo_record_keeps_required_fields(self) -> None:
        record = build_grpo_record(
            {
                "id": "1",
                "question": "Compute 2+2.",
                "final_answer": "4",
                "source": "numinamath",
                "problem_type": "Algebra",
            },
            response_tokens=12,
            reference_solution="Reasoning\n\nFinal answer: \\boxed{4}",
        )
        self.assertEqual(record["id"], "1")
        self.assertEqual(record["final_answer"], "4")
        self.assertEqual(record["response_tokens"], 12)
        self.assertIn("Final answer: \\boxed{...}", record["prompt"])

    def test_reward_functions_follow_contract(self) -> None:
        completions = [
            "Reasoning\n\nFinal answer: \\boxed{4}",
            "Reasoning only",
        ]
        answers = ["4", "4"]
        self.assertEqual(correctness_reward([], completions, answers), [1.0, 0.0])
        self.assertEqual(parse_reward([], completions, answers), [1.0, -1.0])
        self.assertEqual(format_reward([], completions, answers), [1.0, -1.0])

    def test_default_reward_weights_match_result_priority(self) -> None:
        self.assertEqual(default_reward_weights(), [1.0, 0.02, 0.02])

    def test_grpo_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
        self.assertEqual(cfg.loss_type, "dapo")
        self.assertTrue(cfg.mask_truncated_completions)
        self.assertEqual(cfg.max_completion_length, 384)
        self.assertTrue(cfg.use_unsloth)


if __name__ == "__main__":
    unittest.main()
