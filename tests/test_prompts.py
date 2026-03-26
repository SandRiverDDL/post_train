from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.prompts import build_eval_prompt, build_sft_prompt


class PromptTemplateTest(unittest.TestCase):
    def test_build_sft_prompt_uses_strong_format_constraint(self) -> None:
        prompt = build_sft_prompt("Compute 2+2.")
        self.assertIn("Solve the following math problem.", prompt)
        self.assertIn("End your response with a single final answer in the format \\boxed{...}.", prompt)
        self.assertIn("Problem:\nCompute 2+2.", prompt)

    def test_build_eval_prompt_uses_short_template(self) -> None:
        prompt = build_eval_prompt("Compute 2+2.")
        self.assertIn("Compute 2+2.", prompt)
        self.assertIn("Please reason step by step, and put your final answer within \\boxed{}.", prompt)


if __name__ == "__main__":
    unittest.main()
