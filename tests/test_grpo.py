from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.config import GRPOTrainConfig, load_eval_config, load_grpo_config
from rl.grpo import (
    JsonlMetricsCallback,
    build_grpo_prompt,
    build_grpo_record,
    combined_reward,
    default_reward_weights,
    last_reward_stats,
    set_reward_config,
    set_reward_tokenizer,
)
from prepare_grpo_data import build_train_record, infer_source
from train_grpo import init_wandb, resolve_training_model_name


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

    def test_combined_reward_follows_contract(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=True,
        )
        completions = [
            "Reasoning\n\nFinal answer: \\boxed{4}",
            "Reasoning only",
        ]
        answers = ["4", "4"]
        rewards = combined_reward([], completions, answers)
        self.assertAlmostEqual(rewards[0], 1.0196, places=4)
        self.assertAlmostEqual(rewards[1], -0.2002, places=4)
        stats = last_reward_stats()
        self.assertEqual(stats["rewards/correct_rate"], 0.5)
        self.assertEqual(stats["rewards/relaxed_correct_rate"], 0.5)
        self.assertEqual(stats["rewards/strict_correct_rate"], 0.5)
        self.assertEqual(stats["rewards/strict_boxed_rate"], 0.5)
        self.assertEqual(stats["rewards/parse_fail_rate"], 0.5)
        self.assertEqual(stats["rewards/format_rate"], 0.5)
        self.assertEqual(stats["rewards/wrong_rate"], 0.0)

    def test_default_reward_weights_match_result_priority(self) -> None:
        self.assertEqual(default_reward_weights(), [1.0])

    def test_combined_reward_wrong_answer_can_keep_format_bonus(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=True,
        )
        rewards = combined_reward([], ["Try\n\nFinal answer: \\boxed{5}"], ["4"])
        self.assertAlmostEqual(rewards[0], -0.1804, places=4)
        stats = last_reward_stats()
        self.assertEqual(stats["rewards/correct_rate"], 0.0)
        self.assertEqual(stats["rewards/relaxed_correct_rate"], 0.0)
        self.assertEqual(stats["rewards/strict_correct_rate"], 0.0)
        self.assertEqual(stats["rewards/wrong_rate"], 1.0)
        self.assertEqual(stats["rewards/parse_fail_rate"], 0.0)
        self.assertEqual(stats["rewards/format_rate"], 1.0)
        self.assertEqual(stats["rewards/strict_boxed_rate"], 1.0)

    def test_combined_reward_relaxed_correct_gets_small_boxed_bonus(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=True,
        )
        rewards = combined_reward([], ["The answer is 4"], ["4"])
        self.assertAlmostEqual(rewards[0], 0.9996, places=4)
        stats = last_reward_stats()
        self.assertEqual(stats["rewards/relaxed_correct_rate"], 1.0)
        self.assertEqual(stats["rewards/strict_correct_rate"], 0.0)
        self.assertEqual(stats["rewards/strict_boxed_rate"], 0.0)

    def test_combined_reward_can_use_strict_correctness(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=False,
        )
        rewards = combined_reward([], ["The answer is 4"], ["4"])
        self.assertAlmostEqual(rewards[0], -0.2004, places=4)
        stats = last_reward_stats()
        self.assertEqual(stats["rewards/relaxed_correct_rate"], 1.0)
        self.assertEqual(stats["rewards/strict_correct_rate"], 0.0)

    def test_combined_reward_can_disable_format_and_length_rewards(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.0,
            length_coef=0.0,
            use_relaxed_correctness=True,
        )
        rewards = combined_reward([], ["Reasoning\n\nFinal answer: \\boxed{4}"], ["4"])
        self.assertAlmostEqual(rewards[0], 1.0, places=4)

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
        self.assertEqual(cfg.reward_correct, 1.0)
        self.assertEqual(cfg.reward_wrong, -0.2)
        self.assertEqual(cfg.reward_parse_fail, -0.2)
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.02)
        self.assertEqual(cfg.reward_length_coef, 1.0e-4)
        self.assertTrue(cfg.reward_use_relaxed_correctness)
        self.assertEqual(cfg.generation_kwargs, {})
        self.assertTrue(cfg.mask_truncated_completions)
        self.assertEqual(cfg.max_steps, -1)
        self.assertEqual(cfg.max_completion_length, 384)
        self.assertEqual(cfg.lora_rank, 16)
        self.assertEqual(cfg.lora_alpha, 32)
        self.assertTrue(cfg.load_in_4bit)
        self.assertFalse(cfg.fast_inference)
        self.assertEqual(cfg.save_strategy, "epoch")
        self.assertEqual(cfg.save_steps, 50)
        self.assertIsNone(cfg.save_total_limit)
        self.assertEqual(cfg.report_to, "none")
        self.assertEqual(cfg.wandb_mode, "offline")
        self.assertEqual(cfg.wandb_project, "qwen3-math-posttrain")
        self.assertEqual(cfg.wandb_tags, [])

    def test_resolve_training_model_name_prefers_adapter_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=adapter_dir,
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
            self.assertEqual(resolve_training_model_name(cfg), str(adapter_dir))

    def test_resolve_training_model_name_falls_back_to_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "missing_adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
            self.assertEqual(resolve_training_model_name(cfg), "base")

    def test_tiny_grpo_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_tiny.yaml")
        self.assertEqual(cfg.train_dataset, Path("data/grpo/train_grpo_gsm8k_tiny_short.jsonl"))
        self.assertEqual(cfg.max_prompt_length, 256)
        self.assertEqual(cfg.max_completion_length, 256)
        self.assertEqual(cfg.num_generations, 2)
        self.assertEqual(cfg.gradient_accumulation_steps, 4)
        self.assertEqual(cfg.beta, 0.02)
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.02)
        self.assertEqual(cfg.logging_steps, 1)
        self.assertEqual(cfg.save_strategy, "steps")
        self.assertEqual(cfg.save_steps, 25)
        self.assertEqual(cfg.report_to, "wandb")

    def test_ablate_grpo_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate.yaml")
        self.assertEqual(cfg.max_steps, 100)
        self.assertEqual(cfg.save_strategy, "steps")
        self.assertEqual(cfg.save_steps, 25)
        self.assertEqual(cfg.save_total_limit, 4)
        self.assertEqual(cfg.beta, 0.02)
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.02)
        self.assertEqual(cfg.reward_length_coef, 1.0e-4)

    def test_no_format_ablate_grpo_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate_no_format.yaml")
        self.assertEqual(cfg.max_steps, 100)
        self.assertEqual(cfg.beta, 0.02)
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.0)

    def test_no_length_ablate_grpo_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate_no_length.yaml")
        self.assertEqual(cfg.max_steps, 100)
        self.assertEqual(cfg.beta, 0.02)
        self.assertEqual(cfg.reward_length_coef, 0.0)

    def test_correct_only_ablate_grpo_config_loads(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate_correct_only.yaml")
        self.assertEqual(cfg.max_steps, 100)
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.0)

    def test_eval_config_loads(self) -> None:
        cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
        self.assertEqual(cfg.eval_dataset, Path("data/eval/gsm8k_dev200.jsonl"))
        self.assertEqual(cfg.test_dataset, Path("data/eval/math500_test.jsonl"))
        self.assertEqual(cfg.eval_backend, "vllm")
        self.assertEqual(cfg.eval_attn_implementation, "sdpa")
        self.assertEqual(cfg.eval_max_new_tokens, 512)

    def test_grpo_config_supports_base_inheritance(self) -> None:
        cfg = load_grpo_config(ROOT / "configs" / "grpo_ablate_no_format.yaml")
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.train_dataset, Path("data/grpo/train_grpo_gsm8k_1k_middiff.jsonl"))
        self.assertEqual(cfg.reward_strict_boxed_bonus, 0.0)

    def test_infer_source_detects_gsm8k(self) -> None:
        self.assertEqual(infer_source("openai/gsm8k", "auto"), "gsm8k")
        self.assertEqual(
            infer_source(
                "/home/chy/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/cc7b047b6e5bb11b4f1af84efc572db110a51b3c",
                "auto",
            ),
            "gsm8k",
        )
        self.assertEqual(infer_source("nlile/NuminaMath-1.5-RL-Verifiable", "auto"), "numinamath")

    def test_build_train_record_keeps_gsm8k_reasoning_for_grpo(self) -> None:
        row = {
            "question": "Toy question",
            "answer": "We add 2 and 3 to get 5.\n#### 5",
        }
        record = build_train_record(row, 0, "gsm8k")
        self.assertEqual(record["final_answer"], "5")
        self.assertIn("We add 2 and 3 to get 5.", record["solution"])
        self.assertIn("Final answer: \\boxed{5}", record["solution"])
        self.assertEqual(record["problem_type"], "Arithmetic")

    def test_jsonl_metrics_callback_resets_existing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "train_log.jsonl"
            output.write_text("old\n", encoding="utf-8")
            JsonlMetricsCallback(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_jsonl_metrics_callback_writes_reward_stats(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        class DummyState:
            global_step = 3
            epoch = 1.5

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=True,
        )
        combined_reward([], ["Reasoning\n\nFinal answer: \\boxed{4}"], ["4"])
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "train_log.jsonl"
            callback = JsonlMetricsCallback(output)
            callback.on_log(None, DummyState(), None, logs={"reward": 1.0})
            content = output.read_text(encoding="utf-8")
            self.assertIn('"rewards/correct_rate": 1.0', content)
            self.assertIn('"rewards/parse_fail_rate": 0.0', content)

    def test_jsonl_metrics_callback_logs_to_wandb_when_enabled(self) -> None:
        class DummyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

        class DummyState:
            global_step = 5
            epoch = 2.0

        class DummyRun:
            def __init__(self) -> None:
                self.logged = []

            def log(self, record, step=None):
                self.logged.append((record, step))

        set_reward_tokenizer(DummyTokenizer())
        set_reward_config(
            correct=1.0,
            wrong=-0.2,
            parse_fail=-0.2,
            strict_boxed_bonus=0.02,
            length_coef=1e-4,
            use_relaxed_correctness=True,
        )
        combined_reward([], ["Reasoning\n\nFinal answer: \\boxed{4}"], ["4"])
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "train_log.jsonl"
            run = DummyRun()
            callback = JsonlMetricsCallback(output, wandb_run=run)
            callback.on_log(None, DummyState(), None, logs={"reward": 1.0})
            self.assertEqual(len(run.logged), 1)
            record, step = run.logged[0]
            self.assertEqual(step, 5)
            self.assertEqual(record["rewards/correct_rate"], 1.0)
            self.assertEqual(record["reward"], 1.0)

    def test_init_wandb_returns_none_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
            )
            self.assertIsNone(init_wandb(cfg))

    def test_init_wandb_requires_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOTrainConfig(
                base_model_name="base",
                cold_start_model=Path(tmpdir) / "adapter",
                train_dataset=Path(tmpdir) / "train.jsonl",
                report_to="wandb",
            )
            with patch("train_grpo.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "没有安装 wandb"):
                    init_wandb(cfg)


if __name__ == "__main__":
    unittest.main()
