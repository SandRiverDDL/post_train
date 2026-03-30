from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.sft import _compute_sft_loss, _extract_model_inputs, _tokenize_prompt_completion, build_train_dataset


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


class SFTProfitLossTest(unittest.TestCase):
    def test_compute_sft_loss_matches_standard_ce_when_profit_disabled(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            profit_enabled=False,
            profit_threshold=0.1,
        )

        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        expected = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="mean")
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(metrics["valid_tokens"], 2.0)
        self.assertEqual(metrics["kept_tokens"], 2.0)
        self.assertEqual(metrics["filtered_tokens"], 0.0)

    def test_compute_sft_loss_filters_low_probability_tokens_when_profit_enabled(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 3.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            profit_enabled=True,
            profit_threshold=0.1,
        )

        kept_expected = torch.nn.functional.cross_entropy(
            logits[:, :1, :].contiguous().view(-1, logits.size(-1)),
            torch.tensor([1]),
            reduction="mean",
        )
        self.assertAlmostEqual(loss.item(), kept_expected.item(), places=6)
        self.assertEqual(metrics["valid_tokens"], 2.0)
        self.assertEqual(metrics["kept_tokens"], 1.0)
        self.assertEqual(metrics["filtered_tokens"], 1.0)

    def test_compute_sft_loss_returns_zero_when_all_supervised_tokens_are_filtered(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            profit_enabled=True,
            profit_threshold=0.99,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["valid_tokens"], 2.0)
        self.assertEqual(metrics["kept_tokens"], 0.0)
        self.assertEqual(metrics["filtered_tokens"], 2.0)
        self.assertEqual(metrics["empty_batches"], 1.0)

    def test_compute_sft_loss_does_not_require_attention_mask(self) -> None:
        model_inputs = _extract_model_inputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
            }
        )
        self.assertEqual(set(model_inputs.keys()), {"input_ids"})

        logits = torch.tensor(
            [
                [
                    [0.0, 3.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            profit_enabled=True,
            profit_threshold=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["valid_tokens"], 2.0)


if __name__ == "__main__":
    unittest.main()
