from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.prompts import build_eval_prompt, build_math_prompt, build_sft_prompt, render_chat_prompt


class DummyChatTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


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

    def test_build_math_prompt_supports_justrl_style(self) -> None:
        prompt = build_math_prompt("Compute 2+2.", style="justrl_math")
        self.assertEqual(
            prompt,
            "Compute 2+2.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        )

    def test_render_chat_prompt_supports_empty_system_prompt(self) -> None:
        rendered = render_chat_prompt(
            DummyChatTokenizer(),
            "Compute 2+2.\nPlease reason step by step, and put your final answer within \\boxed{}.",
            system_prompt="",
            assistant_prefill=None,
        )

        self.assertIn("<|im_start|>system\n<|im_end|>", rendered)
        self.assertIn("<|im_start|>user\nCompute 2+2.", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
