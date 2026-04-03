from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_grpo_data_config, load_grpo_reward_config, load_grpo_train_config
from post_train.grpo import (
    _aligned_compute_dtype,
    _resolve_train_compute_dtype,
    WandbSmoothingCallback,
    assert_generation_dtype_ready,
    align_output_head_dtypes,
    build_training_args,
    collect_model_dtype_report,
    preflight_check,
)
from post_train.grpo_data import build_grpo_dataset, load_grpo_dataset
from post_train.grpo_data import select_grpo_train_subset
from post_train.grpo_rewards import build_reward_functions


def _write_tmp_grpo_jsonl(rows: list[dict]) -> Path:
    tmp_dir = tempfile.TemporaryDirectory()
    path = Path(tmp_dir.name) / "train.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    if not hasattr(_write_tmp_grpo_jsonl, "_tmp_dirs"):
        _write_tmp_grpo_jsonl._tmp_dirs = []
    _write_tmp_grpo_jsonl._tmp_dirs.append(tmp_dir)
    return path


class GRPOConfigTest(unittest.TestCase):
    def test_load_grpo_data_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "grpo_data.yaml"
            config_path.write_text("", encoding="utf-8")
            cfg = load_grpo_data_config(config_path)

        self.assertEqual(cfg.rd211_dataset, "rd211/Big-Math-RL-Verified-Filtered")
        self.assertEqual(cfg.dataset_split, "train")
        self.assertEqual(cfg.solve_rate_lower, 0.25)
        self.assertEqual(cfg.solve_rate_upper, 0.5)
        self.assertEqual(cfg.anchor_share, 0.3)

    def test_load_grpo_train_config_defaults(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")

        self.assertEqual(cfg.loss_type, "dr_grpo")
        self.assertEqual(cfg.num_generations, 4)
        self.assertEqual(cfg.report_to, "wandb")
        self.assertFalse(cfg.load_in_4bit)
        self.assertEqual(cfg.max_train_samples, 500)
        self.assertEqual(cfg.train_subset_seed, 42)
        self.assertEqual(cfg.train_subset_mode, "fixed_random")
        self.assertEqual(cfg.compute_dtype, "bfloat16")
        self.assertTrue(cfg.mask_truncated_completions)
        self.assertFalse(cfg.wandb_debug_metrics)

    def test_build_training_args_uses_dr_grpo_defaults(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")

        training_args = build_training_args(cfg)

        self.assertEqual(training_args.num_generations, 4)
        self.assertEqual(training_args.loss_type, "dr_grpo")
        self.assertEqual(training_args.scale_rewards, "none")
        self.assertEqual(training_args.importance_sampling_level, "sequence")
        self.assertTrue(training_args.mask_truncated_completions)
        self.assertEqual(training_args.use_vllm, cfg.use_vllm)

    def test_aligned_compute_dtype_uses_bfloat16_when_supported(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")
        cfg = cfg.model_copy(update={"compute_dtype": "bfloat16"})

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = _aligned_compute_dtype(cfg)

        self.assertEqual(str(dtype), "torch.bfloat16")

    def test_resolve_train_compute_dtype_uses_explicit_compute_dtype(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")
        dtype = _resolve_train_compute_dtype(cfg)
        self.assertEqual(str(dtype), "torch.bfloat16")


class GRPODataTest(unittest.TestCase):
    def test_build_grpo_dataset_filters_by_rate_and_exact_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            anchor_path = tmp_path / "anchor.jsonl"
            anchor_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a1", "question": "Anchor only", "final_answer": "5", "meta": {"source": "anchor"}}),
                        json.dumps({"id": "a2", "question": "Seen in rd211", "final_answer": "7", "meta": {"source": "anchor"}}),
                        json.dumps({"id": "a3", "question": "Anchor only", "final_answer": "5", "meta": {"source": "anchor"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "grpo_data.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"anchor_dataset_path: {anchor_path}",
                        f"output_path: {tmp_path / 'train.jsonl'}",
                        f"report_path: {tmp_path / 'report.json'}",
                        "anchor_share: 0.3",
                        "dataset_split: train",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_grpo_data_config(config_path)
            fake_rd211 = [
                {"problem": "Seen in rd211", "answer": "7", "source": "s1", "domain": "alg", "llama8b_solve_rate": 0.3},
                {"problem": "Too easy", "answer": "1", "source": "s1", "domain": "alg", "llama8b_solve_rate": 0.7},
                {"problem": "Fresh rd211", "answer": "9", "source": "s2", "domain": "geo", "llama8b_solve_rate": 0.4},
                {"problem": "Seen in rd211", "answer": "7", "source": "s2", "domain": "geo", "llama8b_solve_rate": 0.31},
            ]

            def fake_load_dataset_rows(dataset_name, *, split, cache_dir=None):
                self.assertEqual(split, "train")
                return fake_rd211

            with patch("post_train.data.load_dataset_rows", side_effect=fake_load_dataset_rows):
                _, result = build_grpo_dataset(cfg)

            rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result["report"]["rd211_count"], 2)
        self.assertEqual(result["report"]["anchor_count"], 1)
        self.assertEqual(result["report"]["cross_pool_duplicate_count"], 1)
        self.assertTrue(result["report"]["anchor_is_bottleneck"])
        self.assertEqual(result["report"]["rd211_dropped_for_ratio"], 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["question"] for row in rows}), 3)
        self.assertTrue(any(row["question"] == "Anchor only" for row in rows))

    def test_load_grpo_dataset_requires_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "1",
                        "prompt": "Solve",
                        "question": "1+1?",
                        "final_answer": "2",
                        "meta": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = load_grpo_dataset(path)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["final_answer"], "2")

    def test_select_grpo_train_subset_keeps_full_dataset_when_limit_is_none(self) -> None:
        dataset = load_grpo_dataset(_write_tmp_grpo_jsonl([{"id": str(i), "prompt": "p", "question": f"q{i}", "final_answer": "a", "meta": {}} for i in range(3)]))
        subset = select_grpo_train_subset(dataset, max_train_samples=None, seed=42, mode="fixed_random")
        self.assertEqual(len(subset), 3)

    def test_select_grpo_train_subset_fixed_random_is_stable(self) -> None:
        dataset = load_grpo_dataset(_write_tmp_grpo_jsonl([{"id": str(i), "prompt": "p", "question": f"q{i}", "final_answer": "a", "meta": {}} for i in range(10)]))
        subset_a = select_grpo_train_subset(dataset, max_train_samples=5, seed=42, mode="fixed_random")
        subset_b = select_grpo_train_subset(dataset, max_train_samples=5, seed=42, mode="fixed_random")
        self.assertEqual(subset_a["id"], subset_b["id"])

    def test_select_grpo_train_subset_head_takes_prefix(self) -> None:
        dataset = load_grpo_dataset(_write_tmp_grpo_jsonl([{"id": str(i), "prompt": "p", "question": f"q{i}", "final_answer": "a", "meta": {}} for i in range(10)]))
        subset = select_grpo_train_subset(dataset, max_train_samples=3, seed=42, mode="head")
        self.assertEqual(subset["id"], ["0", "1", "2"])


class GRPORewardTest(unittest.TestCase):
    def test_build_reward_functions_scores_correct_and_parse_fail(self) -> None:
        cfg = load_grpo_reward_config("configs/grpo/reward.yaml")
        reward_funcs = build_reward_functions(cfg)
        correctness_reward = reward_funcs[0]
        parse_penalty_reward = reward_funcs[1]

        completions = ["解答...\n\\boxed{2}", "没有框起来的答案 3"]
        final_answers = ["2", "3"]

        self.assertEqual(correctness_reward(completions, final_answers), [1.0, 0.0])
        self.assertEqual(parse_penalty_reward(completions, final_answers), [0.0, -0.5])


class GRPOPreflightTest(unittest.TestCase):
    def test_preflight_requires_gpu(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "需要可用 GPU"):
                preflight_check(cfg)

    def test_preflight_rejects_incompatible_vllm_version(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_4090.yaml")

        def fake_version(name: str) -> str:
            return {"trl": "0.24.0", "vllm": "0.12.0"}[name]

        with patch("torch.cuda.is_available", return_value=True), patch("post_train.grpo.metadata.version", side_effect=fake_version):
            with self.assertRaisesRegex(RuntimeError, "只支持 vllm==0.10.2"):
                preflight_check(cfg)

    def test_preflight_allows_newer_trl_with_current_vllm(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_4090.yaml")

        def fake_version(name: str) -> str:
            return {"trl": "0.25.1", "vllm": "0.12.0"}[name]

        with patch("torch.cuda.is_available", return_value=True), patch("post_train.grpo.metadata.version", side_effect=fake_version):
            preflight_check(cfg)


class GRPODTypeTest(unittest.TestCase):
    def test_align_output_head_dtypes_only_updates_output_head(self) -> None:
        import torch

        class DummyEmbedding:
            def __init__(self, dtype):
                self.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))

        class DummyModel:
            def __init__(self) -> None:
                self.output_embeddings = DummyEmbedding(torch.float32)
                self.input_embeddings = DummyEmbedding(torch.float32)

            def get_output_embeddings(self):
                return self.output_embeddings

            def get_input_embeddings(self):
                return self.input_embeddings

        model = DummyModel()
        touched_paths = align_output_head_dtypes(model, target_dtype=torch.bfloat16)

        self.assertEqual(model.output_embeddings.weight.dtype, torch.bfloat16)
        self.assertEqual(model.input_embeddings.weight.dtype, torch.float32)
        self.assertEqual(touched_paths, ["model.get_output_embeddings()"])

    def test_align_output_head_dtypes_reaches_deep_base_model_lm_head(self) -> None:
        import torch

        class DummyHead:
            def __init__(self, dtype):
                self.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))

        class WrappedBase:
            def __init__(self):
                self.lm_head = DummyHead(torch.float32)

        class WrappedLora:
            def __init__(self):
                self.model = WrappedBase()

        class WrappedPeft:
            def __init__(self):
                self.base_model = WrappedLora()

        model = WrappedPeft()
        touched_paths = align_output_head_dtypes(model, target_dtype=torch.bfloat16)
        report = collect_model_dtype_report(model)

        self.assertIn("model.base_model.model.lm_head", touched_paths)
        self.assertEqual(model.base_model.model.lm_head.weight.dtype, torch.bfloat16)
        self.assertEqual(report["model.base_model.model.lm_head.weight_dtype"], "torch.bfloat16")

    def test_assert_generation_dtype_ready_rejects_mismatched_output_head(self) -> None:
        import torch

        class DummyEmbedding:
            def __init__(self, dtype):
                self.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))

        class DummyModel:
            def __init__(self) -> None:
                self.output_embeddings = DummyEmbedding(torch.float32)

            def get_output_embeddings(self):
                return self.output_embeddings

        with self.assertRaisesRegex(RuntimeError, "dtype 检查失败"):
            assert_generation_dtype_ready(DummyModel(), target_dtype=torch.bfloat16)

    def test_assert_generation_dtype_ready_rejects_when_no_output_head_found(self) -> None:
        import torch

        class DummyEmbedding:
            def __init__(self, dtype):
                self.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))

        class DummyModel:
            def __init__(self) -> None:
                self.input_embeddings = DummyEmbedding(torch.float32)

            def get_input_embeddings(self):
                return self.input_embeddings

        with self.assertRaisesRegex(RuntimeError, "未找到任何可用于生成的 output head"):
            assert_generation_dtype_ready(DummyModel(), target_dtype=torch.bfloat16)


class GRPOWandbTest(unittest.TestCase):
    def test_wandb_callback_logs_only_main_metrics_by_default(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")
        callback = WandbSmoothingCallback(cfg)

        self.assertTrue(callback._should_log("reward/correctness"))
        self.assertTrue(callback._should_log("reward/parse_penalty"))
        self.assertTrue(callback._should_log("completions/mean_length"))
        self.assertFalse(callback._should_log("loss"))
        self.assertFalse(callback._should_log("reward_std"))

    def test_wandb_callback_can_enable_debug_metrics(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")
        cfg = cfg.model_copy(update={"wandb_debug_metrics": True})
        callback = WandbSmoothingCallback(cfg)

        self.assertTrue(callback._should_log("loss"))
        self.assertTrue(callback._should_log("reward_std"))
        self.assertTrue(callback._should_log("frac_reward_zero_std"))


if __name__ == "__main__":
    unittest.main()
