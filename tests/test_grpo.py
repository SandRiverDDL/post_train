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
from post_train.grpo import _aligned_compute_dtype, build_training_args, preflight_check
from post_train.grpo_data import build_grpo_dataset, load_grpo_dataset
from post_train.grpo_rewards import build_reward_functions


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
        self.assertFalse(cfg.use_vllm)
        self.assertEqual(cfg.report_to, "wandb")

    def test_build_training_args_uses_dr_grpo_defaults(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")

        training_args = build_training_args(cfg)

        self.assertEqual(training_args.num_generations, 4)
        self.assertEqual(training_args.loss_type, "dr_grpo")
        self.assertEqual(training_args.scale_rewards, "none")
        self.assertEqual(training_args.importance_sampling_level, "sequence")
        self.assertFalse(training_args.use_vllm)

    def test_aligned_compute_dtype_uses_bfloat16_when_supported(self) -> None:
        cfg = load_grpo_train_config("configs/grpo/train_cheap.yaml")

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = _aligned_compute_dtype(cfg)

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


if __name__ == "__main__":
    unittest.main()
