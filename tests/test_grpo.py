from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.config import GRPOTrainConfig
from rl.grpo import (
    align_lm_head_dtype,
    build_grpo_prompt,
    build_grpo_record,
    correctness_reward,
    default_reward_weights,
    ensure_trl_model_compat,
    ensure_trl_vllm_import_compat,
    find_unquantized_linear_modules,
    format_reward,
    model_uses_kbit_quantization,
    parse_reward,
    stabilize_unquantized_kbit_linears,
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

    def test_model_uses_kbit_quantization_checks_4bit_and_8bit_flags(self) -> None:
        class DummyModel:
            is_loaded_in_4bit = True

        self.assertTrue(model_uses_kbit_quantization(DummyModel()))
        self.assertFalse(model_uses_kbit_quantization(object()))

    def test_find_unquantized_linear_modules_collects_linear_paths(self) -> None:
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.block = nn.Sequential(nn.ReLU(), nn.Linear(4, 2))

        names = find_unquantized_linear_modules(DummyModel())
        self.assertEqual(names, ["linear", "block.1"])

    def test_stabilize_unquantized_kbit_linears_only_upcasts_non_lm_head(self) -> None:
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.is_loaded_in_4bit = True
                self.block = nn.Linear(4, 4, dtype=torch.bfloat16)
                self.lm_head = nn.Linear(4, 4, dtype=torch.bfloat16)

        model = DummyModel()
        stabilized_model, names = stabilize_unquantized_kbit_linears(model)
        self.assertIs(stabilized_model, model)
        self.assertEqual(names, ["block"])
        self.assertEqual(model.block.weight.dtype, torch.float32)
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)

    def test_align_lm_head_dtype_only_updates_plain_linear_head(self) -> None:
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = nn.Linear(4, 4, dtype=torch.float32)

        model = DummyModel()
        changed = align_lm_head_dtype(model, torch.bfloat16)
        self.assertTrue(changed)
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)

    def test_ensure_trl_vllm_import_compat_adds_alias_when_vllm_not_used(self) -> None:
        sampling_params = SimpleNamespace(StructuredOutputsParams=object)
        with patch("importlib.import_module", return_value=sampling_params):
            ensure_trl_vllm_import_compat(use_vllm=False)
        self.assertIs(sampling_params.GuidedDecodingParams, object)

    def test_ensure_trl_vllm_import_compat_rejects_real_vllm_path_with_old_trl_api(self) -> None:
        sampling_params = SimpleNamespace(StructuredOutputsParams=object)
        with patch("importlib.import_module", return_value=sampling_params):
            with self.assertRaisesRegex(RuntimeError, "GuidedDecodingParams"):
                ensure_trl_vllm_import_compat(use_vllm=True)

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
        self.assertTrue(cfg.use_unsloth)

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
