from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.config import GRPOTrainConfig
from rl.grpo import (
    build_grpo_prompt,
    build_grpo_record,
    correctness_reward,
    default_reward_weights,
    ensure_trl_model_compat,
    ensure_trl_vllm_import_compat,
    format_reward,
    parse_reward,
)
from train_grpo import resolve_precision_flags


class GRPOUtilsTest(unittest.TestCase):
    def test_ensure_trl_model_compat_adds_missing_trl_fields(self) -> None:
        class DummyBase:
            pass

        class DummyModel:
            def __init__(self) -> None:
                self.base_model = DummyBase()

        model = ensure_trl_model_compat(DummyModel())
        self.assertEqual(model.warnings_issued, {})
        self.assertIs(model.base_model.warnings_issued, model.warnings_issued)
        self.assertTrue(callable(model.add_model_tags))
        self.assertTrue(callable(model.base_model.add_model_tags))

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

    def test_ensure_trl_vllm_import_compat_adds_alias_for_vllm_modes(self) -> None:
        sampling_params = SimpleNamespace(StructuredOutputsParams=object)
        with patch("importlib.import_module", return_value=sampling_params):
            applied = ensure_trl_vllm_import_compat(use_vllm=True, vllm_mode="colocate")
        self.assertTrue(applied)
        self.assertIs(sampling_params.GuidedDecodingParams, object)

    def test_grpo_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
        self.assertEqual(cfg.loss_type, "dapo")
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.generation_kwargs, {})
        self.assertIsNone(cfg.bf16)
        self.assertFalse(cfg.fp16)
        self.assertTrue(cfg.mask_truncated_completions)
        self.assertEqual(cfg.max_completion_length, 384)
        self.assertEqual(cfg.attn_implementation, "sdpa")
        self.assertFalse(cfg.use_vllm)
        self.assertEqual(cfg.vllm_mode, "server")
        self.assertEqual(cfg.vllm_server_host, "127.0.0.1")
        self.assertEqual(cfg.vllm_server_port, 8000)
        self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
        self.assertEqual(cfg.vllm_data_parallel_size, 1)

    def test_compute_dtype_matches_device_capability_rule(self) -> None:
        expected = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.assertIn(expected, (torch.bfloat16, torch.float16))

    def test_resolve_precision_flags_do_not_auto_enable_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
        use_bf16, use_fp16, _ = resolve_precision_flags(cfg)
        self.assertFalse(use_bf16)
        self.assertFalse(use_fp16)


if __name__ == "__main__":
    unittest.main()
