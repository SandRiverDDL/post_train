from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.sft import _tokenize_prompt_completion, build_train_dataset


class BoundaryMergingTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if text == "Prompt.":
            return {"input_ids": [1, 13]}
        if text == "Let answer":
            return {"input_ids": [2, 3]}
        if text == "Prompt.Let answer":
            return {"input_ids": [1, 1214, 3]}
        raise AssertionError(f"未预期的输入：{text!r}")


class WhitespaceTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        pieces = [piece for piece in text.split() if piece]
        return {"input_ids": list(range(1, len(pieces) + 1))}


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
        self.assertEqual(encoded["length"], 4)

    def test_build_train_dataset_adds_length_column(self) -> None:
        tokenizer = WhitespaceTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "train.jsonl"
            dataset_path.write_text(
                '{"id":"1","question":"What is 1+1?","solution":"Reasoning\\\\n\\\\n\\\\boxed{2}","final_answer":"2"}\n',
                encoding="utf-8",
            )
            dataset = build_train_dataset(dataset_path, tokenizer, max_length=64)

        row = dataset[0]
        self.assertIn("length", row)
        self.assertEqual(row["length"], len(row["input_ids"]))


if __name__ == "__main__":
    unittest.main()
