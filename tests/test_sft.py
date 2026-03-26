from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.sft import _tokenize_prompt_completion


class BoundaryMergingTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if text == "Prompt.":
            return {"input_ids": [1, 13]}
        if text == "Let answer":
            return {"input_ids": [2, 3]}
        if text == "Prompt.Let answer":
            return {"input_ids": [1, 1214, 3]}
        raise AssertionError(f"未预期的输入：{text!r}")


class SFTTokenizationTest(unittest.TestCase):
    def test_tokenize_prompt_completion_handles_boundary_merge(self) -> None:
        tokenizer = BoundaryMergingTokenizer()

        encoded = _tokenize_prompt_completion(
            tokenizer,
            prompt="Prompt.",
            completion="Let answer",
            max_length=16,
        )

        self.assertEqual(encoded["input_ids"], [1, 13, 2, 3])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1])
        self.assertEqual(encoded["labels"], [-100, -100, 2, 3])


if __name__ == "__main__":
    unittest.main()
