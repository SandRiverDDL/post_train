from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.sft import (
    LightningOPDDataCollator,
    _compute_lightning_opd_loss,
    _compute_sft_loss,
    _extract_model_inputs,
    _tokenize_prompt_completion,
    build_lightning_opd_dataset,
    build_train_dataset,
)


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


class WhitespaceTokenizerWithEos(WhitespaceTokenizer):
    eos_token_id = 99


class RecordingChatTokenizer:
    chat_template = "dummy"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        self.texts.append(text)
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

    def test_tokenize_prompt_completion_appends_eos_to_completion_labels(self) -> None:
        tokenizer = WhitespaceTokenizerWithEos()

        encoded = _tokenize_prompt_completion(
            tokenizer,
            prompt="Prompt text",
            completion="Final answer",
            max_length=16,
        )

        self.assertEqual(encoded["input_ids"], [1, 2, 1, 2, 99])
        self.assertEqual(encoded["labels"], [-100, -100, 1, 2, 99])

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

    def test_build_train_dataset_can_render_justrl_chat_prompt(self) -> None:
        tokenizer = RecordingChatTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "train.jsonl"
            dataset_path.write_text(
                '{"id":"1","question":"What is 1+1?","solution":"Reasoning\\\\n\\\\n\\\\boxed{2}","final_answer":"2"}\n',
                encoding="utf-8",
            )
            build_train_dataset(
                dataset_path,
                tokenizer,
                max_length=64,
                prompt_style="justrl_math",
                use_chat_template=True,
                system_prompt="",
            )

        rendered_prompt = tokenizer.texts[0]
        self.assertIn("<|im_start|>system\n<|im_end|>", rendered_prompt)
        self.assertIn("What is 1+1?", rendered_prompt)
        self.assertIn("Please reason step by step, and put your final answer within \\boxed{}.", rendered_prompt)
        self.assertTrue(rendered_prompt.endswith("<|im_start|>assistant\n"))


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
            loss_mode="standard",
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
            loss_mode="standard",
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

    def test_compute_sft_loss_supports_dft_gold_probability_weighting(self) -> None:
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
            loss_mode="dft",
            profit_enabled=False,
            profit_threshold=0.1,
        )

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        gold_prob = torch.exp(-token_nll.detach())
        expected = (gold_prob * token_nll).mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertLess(loss.item(), token_nll.mean().item())
        self.assertEqual(metrics["valid_tokens"], 2.0)
        self.assertEqual(metrics["kept_tokens"], 2.0)
        self.assertEqual(metrics["filtered_tokens"], 0.0)
        self.assertAlmostEqual(metrics["dft_loss_scale"], gold_prob.mean().item(), places=6)

    def test_compute_sft_loss_dft_returns_zero_for_empty_batch(self) -> None:
        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        labels = torch.full((1, 3), -100, dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            loss_mode="dft",
            profit_enabled=False,
            profit_threshold=0.1,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["valid_tokens"], 0.0)
        self.assertEqual(metrics["empty_batches"], 1.0)

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
            loss_mode="standard",
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
            loss_mode="standard",
            profit_enabled=True,
            profit_threshold=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["valid_tokens"], 2.0)

    def test_compute_sft_loss_supports_opsft_batch_max_normalization(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [-100, 1, 1, 1],
                [-100, 1, -100, -100],
            ],
            dtype=torch.long,
        )

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            loss_mode="opsft",
            profit_enabled=False,
            profit_threshold=0.1,
        )

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        per_token = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        sample_sums = torch.tensor(
            [
                float(per_token[0, :].sum().item()),
                float(per_token[1, 0].item()),
            ]
        )
        expected = ((sample_sums[0] / 3.0) + (sample_sums[1] / 3.0)) / 2.0
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(metrics["opsft_batch_max_completion_tokens"], 3.0)
        self.assertEqual(metrics["opsft_effective_sample_count"], 2.0)

    def test_compute_sft_loss_opsft_returns_zero_for_empty_batch(self) -> None:
        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        labels = torch.full((1, 3), -100, dtype=torch.long)

        loss, metrics = _compute_sft_loss(
            logits,
            labels,
            loss_mode="opsft",
            profit_enabled=False,
            profit_threshold=0.1,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["opsft_effective_sample_count"], 0.0)
        self.assertEqual(metrics["opsft_zero_completion_batches"], 1.0)


class LightningOPDTest(unittest.TestCase):
    def test_build_lightning_opd_dataset_aligns_teacher_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "train.jsonl"
            dataset_path.write_text(
                (
                    '{"id":"1","input_ids":[10,11,12],"response_mask":[0,1,1],'
                    '"teacher_token_logprobs":[-0.1,-0.2],'
                    '"teacher_topk_token_ids":[[11,1],[12,2]],'
                    '"teacher_topk_logprobs":[[-0.1,-1.0],[-0.2,-1.2]]}\n'
                ),
                encoding="utf-8",
            )

            dataset = build_lightning_opd_dataset(dataset_path, max_length=8, distill_top_k=2)
            row = dataset[0]

        self.assertEqual(row["labels"], [-100, 11, 12])
        self.assertEqual(row["teacher_token_logprobs"], [0.0, -0.1, -0.2])
        self.assertEqual(row["teacher_topk_token_ids"][1], [11, 1])
        self.assertEqual(row["teacher_topk_mask"][0], [0, 0])

    def test_build_lightning_opd_dataset_rejects_missing_requested_topk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "train.jsonl"
            dataset_path.write_text(
                (
                    '{"id":"1","input_ids":[10,11],"response_mask":[0,1],'
                    '"teacher_token_logprobs":[-0.1],'
                    '"teacher_topk_token_ids":[[11]],'
                    '"teacher_topk_logprobs":[[-0.1]]}\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "distill_top_k=2"):
                build_lightning_opd_dataset(dataset_path, max_length=8, distill_top_k=2)

    def test_lightning_opd_collator_pads_nested_topk_fields(self) -> None:
        features = [
            {
                "input_ids": [10, 11],
                "attention_mask": [1, 1],
                "labels": [-100, 11],
                "response_mask": [0, 1],
                "teacher_token_logprobs": [0.0, -0.1],
                "teacher_topk_token_ids": [[0, 0], [11, 1]],
                "teacher_topk_logprobs": [[0.0, 0.0], [-0.1, -1.0]],
                "teacher_topk_mask": [[0, 0], [1, 1]],
            },
            {
                "input_ids": [20],
                "attention_mask": [1],
                "labels": [-100],
                "response_mask": [0],
                "teacher_token_logprobs": [0.0],
                "teacher_topk_token_ids": [[0, 0]],
                "teacher_topk_logprobs": [[0.0, 0.0]],
                "teacher_topk_mask": [[0, 0]],
            },
        ]

        batch = LightningOPDDataCollator(pad_token_id=99)(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2))
        self.assertEqual(tuple(batch["teacher_topk_token_ids"].shape), (2, 2, 2))
        self.assertEqual(batch["input_ids"][1, 1].item(), 99)

    def test_compute_lightning_opd_loss_baseline_uses_trajectory_teacher_logprob(self) -> None:
        logits = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]], dtype=torch.float32)
        input_ids = torch.tensor([[0, 1]], dtype=torch.long)
        response_mask = torch.tensor([[False, True]])
        teacher_logprobs = torch.tensor([[0.0, -0.2]], dtype=torch.float32)
        topk_ids = torch.zeros((1, 2, 1), dtype=torch.long)
        topk_logprobs = torch.zeros((1, 2, 1), dtype=torch.float32)
        topk_mask = torch.zeros((1, 2, 1), dtype=torch.bool)

        loss, metrics = _compute_lightning_opd_loss(
            logits,
            input_ids,
            response_mask,
            teacher_logprobs,
            topk_ids,
            topk_logprobs,
            topk_mask,
            distill_top_k=1,
            topk_kd_weight=0.0,
            opd_weight=1.0,
        )

        student_logprob = torch.log_softmax(logits[:, 0, :], dim=-1)[0, 1]
        expected = -((-0.2 - student_logprob.detach()) * student_logprob)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(metrics["valid_tokens"], 1.0)
        self.assertEqual(metrics["topk_kd_loss"], 0.0)

    def test_compute_lightning_opd_loss_adds_topk_auxiliary(self) -> None:
        logits = torch.tensor([[[0.0, 2.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32)
        input_ids = torch.tensor([[0, 1]], dtype=torch.long)
        response_mask = torch.tensor([[False, True]])
        teacher_logprobs = torch.tensor([[0.0, -0.2]], dtype=torch.float32)
        topk_ids = torch.tensor([[[0, 0], [1, 2]]], dtype=torch.long)
        topk_logprobs = torch.tensor([[[0.0, 0.0], [-0.1, -1.1]]], dtype=torch.float32)
        topk_mask = torch.tensor([[[False, False], [True, True]]])

        loss, metrics = _compute_lightning_opd_loss(
            logits,
            input_ids,
            response_mask,
            teacher_logprobs,
            topk_ids,
            topk_logprobs,
            topk_mask,
            distill_top_k=2,
            topk_kd_weight=0.5,
            opd_weight=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["topk_kd_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
