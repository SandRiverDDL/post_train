from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.config import EvalConfig, GRPOTrainConfig, SFTConfig, load_config, load_eval_config, load_grpo_config
from rl.data import format_protocol_prompt
from score_grpo_candidates import build_cache_key


class PromptV2Test(unittest.TestCase):
    def test_protocol_prompt_v1_keeps_current_contract(self) -> None:
        prompt = format_protocol_prompt("Compute 2+2.", prompt_version="v1")
        self.assertIn("Question:\nCompute 2+2.", prompt)
        self.assertIn("要求：", prompt)
        self.assertIn("Final answer: \\boxed{...}", prompt)

    def test_protocol_prompt_v2_uses_english_instruction_first(self) -> None:
        prompt = format_protocol_prompt("Compute 2+2.", prompt_version="v2")
        self.assertTrue(prompt.startswith("Please reason step by step, and put your final answer within \\boxed{}."))
        self.assertIn("\n\nQuestion:\nCompute 2+2.\n\nSolution:\n", prompt)
        self.assertNotIn("要求：", prompt)

    def test_sft_config_defaults_prompt_version_to_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SFTConfig(
                model_name="Qwen/Qwen3-0.6B-Base",
                train_dataset=Path(tmpdir) / "train.jsonl",
                eval_dataset=Path(tmpdir) / "eval.jsonl",
            )
        self.assertEqual(cfg.prompt_version, "v1")

    def test_eval_config_defaults_prompt_version_to_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvalConfig(
                model_name="Qwen/Qwen3-0.6B-Base",
                eval_dataset=Path(tmpdir) / "eval.jsonl",
            )
        self.assertEqual(cfg.prompt_version, "v1")

    def test_grpo_config_defaults_prompt_version_to_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="Qwen/Qwen3-0.6B-Base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
        self.assertEqual(cfg.prompt_version, "v1")

    def test_score_cache_key_includes_prompt_version(self) -> None:
        row = {
            "id": "1",
            "question": "Compute 2+2.",
            "final_answer": "4",
        }
        prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\nQuestion:\nCompute 2+2.\n\nSolution:\n"
        cache_key = build_cache_key(
            row,
            prompt=prompt,
            model_id="base=Qwen/Qwen3-0.6B-Base|adapter=outputs/sft-qwen3-0.6b-v2",
            sampling={"num_samples_per_problem": 4, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512},
            prompt_version="protocol_prompt_v2",
        )
        self.assertIn('"score_prompt_version": "protocol_prompt_v2"', cache_key)

    def test_sft_qwen3_0_6b_v2_config_loads(self) -> None:
        cfg = load_config(ROOT / "configs" / "sft_qwen3_0.6b_v2.yaml")
        self.assertEqual(cfg.model_name, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(cfg.output_dir, Path("outputs/sft-qwen3-0.6b-v2"))
        self.assertEqual(cfg.prompt_version, "v2")

    def test_eval_qwen3_0_6b_v2_config_loads(self) -> None:
        cfg = load_eval_config(ROOT / "configs" / "eval_qwen3_0.6b_v2.yaml")
        self.assertEqual(cfg.model_name, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(cfg.prompt_version, "v2")

    def test_grpo_qwen3_0_6b_v2_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_base_qwen3_0.6b_v2.yaml")
        self.assertEqual(cfg.base_model_name, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(cfg.cold_start_model, Path("outputs/sft-qwen3-0.6b-v2"))
        self.assertEqual(cfg.output_dir, Path("outputs/grpo-qwen3-0.6b-v2"))
        self.assertEqual(cfg.prompt_version, "v2")

    def test_grpo_ablate_correct_only_qwen3_0_6b_v2_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate_correct_only_qwen3_0.6b_v2.yaml")
        self.assertEqual(cfg.base_model_name, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(cfg.cold_start_model, Path("outputs/sft-qwen3-0.6b-v2"))
        self.assertEqual(cfg.output_dir, Path("outputs/grpo-qwen3-0.6b-ablate-correct-only-v2"))
        self.assertEqual(cfg.prompt_version, "v2")
        self.assertEqual(cfg.max_steps, 50)


if __name__ == "__main__":
    unittest.main()
