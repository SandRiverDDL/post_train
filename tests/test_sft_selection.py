from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config
from post_train.sft_selection import checkpoint_step, evaluate_checkpoint, run_checkpoint_selection, scan_checkpoint_dirs, select_best_checkpoint


class SFTSelectionTest(unittest.TestCase):
    def test_load_sft_config_supports_checkpoint_save_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "stage1.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_sft",
                        "epochs: 2",
                        "loss_mode: opsft",
                        "profit_enabled: true",
                        "profit_threshold: 0.2",
                        "save_strategy: steps",
                        "save_steps: 25",
                        "save_total_limit: 12",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "opsft 与 profit_enabled 不能同时开启"):
                load_sft_config(config_path)

    def test_load_sft_config_supports_opsft_loss_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "opsft.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_sft",
                        "epochs: 1",
                        "loss_mode: opsft",
                        "profit_enabled: false",
                        'save_strategy: "no"',
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.epochs, 1)
        self.assertEqual(cfg.loss_mode, "opsft")
        self.assertFalse(cfg.profit_enabled)

    def test_load_sft_config_supports_dft_loss_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "dft.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_dft",
                        "epochs: 1",
                        "loss_mode: dft",
                        "profit_enabled: false",
                        'save_strategy: "no"',
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.loss_mode, "dft")
        self.assertFalse(cfg.profit_enabled)

    def test_load_sft_config_supports_lora_quantization_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bf16_lora.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_sft",
                        "quantization: bf16_lora",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.quantization, "bf16_lora")

    def test_load_sft_config_defaults_to_qlora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "default.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_sft",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.quantization, "qlora_4bit")

    def test_load_sft_config_rejects_dft_with_profit_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "dft_profit.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_dft",
                        "loss_mode: dft",
                        "profit_enabled: true",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dft 与 profit_enabled 不能同时开启"):
                load_sft_config(config_path)

    def test_load_sft_config_supports_asft_topk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "asft.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_asft",
                        "loss_mode: asft_topk",
                        "asft_top_k: 32",
                        "asft_kl_weight: 0.03",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_sft_config(config_path)

        self.assertEqual(cfg.loss_mode, "asft_topk")
        self.assertEqual(cfg.asft_top_k, 32)
        self.assertEqual(cfg.asft_kl_weight, 0.03)

    def test_load_sft_config_rejects_asft_topk_with_profit_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "asft_profit.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: /tmp/model",
                        "train_dataset: data/stage1/train.jsonl",
                        "output_dir: outputs/stage1_asft",
                        "loss_mode: asft_topk",
                        "profit_enabled: true",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "asft_topk 与 profit_enabled 不能同时开启"):
                load_sft_config(config_path)

    def test_scan_checkpoint_dirs_sorts_by_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "checkpoint-100").mkdir()
            (output_dir / "checkpoint-25").mkdir()
            (output_dir / "checkpoint-50").mkdir()
            (output_dir / "logs").mkdir()

            checkpoints = scan_checkpoint_dirs(output_dir)

        self.assertEqual([path.name for path in checkpoints], ["checkpoint-25", "checkpoint-50", "checkpoint-100"])
        self.assertEqual(checkpoint_step(checkpoints[-1]), 100)

    def test_select_best_checkpoint_prefers_accuracy_then_format(self) -> None:
        records = [
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-25",
                "global_step": 25,
                "metrics": {
                    "normalized_accuracy": 0.80,
                    "boxed_rate": 0.95,
                    "parse_success_rate": 0.95,
                },
            },
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-50",
                "global_step": 50,
                "metrics": {
                    "normalized_accuracy": 0.80,
                    "boxed_rate": 0.98,
                    "parse_success_rate": 0.97,
                },
            },
            {
                "checkpoint_path": "outputs/stage1_sft/checkpoint-75",
                "global_step": 75,
                "metrics": {
                    "normalized_accuracy": 0.78,
                    "boxed_rate": 1.0,
                    "parse_success_rate": 1.0,
                },
            },
        ]

        best = select_best_checkpoint(records)

        self.assertEqual(best["checkpoint_path"], "outputs/stage1_sft/checkpoint-50")

    def test_evaluate_checkpoint_preserves_model_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoint-10"
            checkpoint_dir.mkdir()
            sentinel_runner = object()
            captured: dict[str, object] = {}

            class FakeEvalRunner:
                def __init__(self, eval_cfg, *, model_name=None, backend_runner=None) -> None:
                    captured["eval_cfg"] = eval_cfg
                    captured["model_name"] = model_name
                    captured["backend_runner"] = backend_runner

                def run_task(self, task, overrides=None):
                    captured["task"] = task
                    captured["overrides"] = overrides
                    return {
                        "result": {"metrics": {"normalized_accuracy": 0.25}},
                        "result_path": "/tmp/result.json",
                        "raw_result_path": "/tmp/raw.json",
                        "model_resolution": {
                            "enable_lora": True,
                            "lora_local_path": str(checkpoint_dir),
                            "pretrained": "/tmp/base",
                        },
                    }

            with patch("post_train.sft_selection.EvalRunner", FakeEvalRunner):
                record = evaluate_checkpoint(
                    checkpoint_path=checkpoint_dir,
                    dataset_path=Path(tmp_dir) / "dev.jsonl",
                    eval_cfg=object(),
                    batch_size=1,
                    max_new_tokens=128,
                    limit=1,
                    output_dir=Path(tmp_dir) / "dev_eval",
                    runner=sentinel_runner,
                )

        self.assertTrue(record["model_resolution"]["enable_lora"])
        self.assertEqual(record["model_resolution"]["pretrained"], "/tmp/base")
        self.assertIs(captured["backend_runner"], sentinel_runner)

    def test_run_checkpoint_selection_reuses_single_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_output_dir = Path(tmp_dir) / "outputs"
            checkpoint_a = train_output_dir / "checkpoint-10"
            checkpoint_b = train_output_dir / "checkpoint-20"
            for checkpoint_dir in (checkpoint_a, checkpoint_b):
                checkpoint_dir.mkdir(parents=True)
            eval_cfg = type(
                "EvalCfg",
                (),
                {
                    "backend": "vllm",
                    "batch_size": 2,
                    "max_new_tokens": 256,
                    "model_name": "/tmp/base-model",
                    "max_seq_length": 1536,
                    "device": "cuda",
                    "attn_implementation": "sdpa",
                    "gpu_memory_utilization": 0.7,
                    "max_lora_rank": 64,
                },
            )()
            class RunnerStub:
                def close(self) -> None:
                    return None

            runner_instance = RunnerStub()

            def fake_evaluate_checkpoint(**kwargs):
                checkpoint_path = Path(kwargs["checkpoint_path"])
                self.assertIs(kwargs["runner"], runner_instance)
                return {
                    "checkpoint_path": str(checkpoint_path),
                    "global_step": checkpoint_step(checkpoint_path),
                    "metrics": {
                        "normalized_accuracy": 0.5 + checkpoint_step(checkpoint_path) / 1000,
                        "boxed_rate": 0.9,
                        "parse_success_rate": 0.9,
                    },
                    "result_path": f"/tmp/{checkpoint_path.name}/result.json",
                    "raw_result_path": f"/tmp/{checkpoint_path.name}/raw.json",
                    "model_resolution": {"enable_lora": True, "lora_local_path": str(checkpoint_path), "pretrained": "/tmp/base-model"},
                }

            with patch("post_train.sft_selection.resolve_model_args", return_value={"pretrained": "/tmp/base-model", "gpu_memory_utilization": 0.7, "max_length": 1536, "lora_local_path": str(checkpoint_a), "max_lora_rank": 64}) as mock_resolve, patch("post_train.sft_selection.VLLMRunner", return_value=runner_instance) as mock_runner_cls, patch("post_train.sft_selection.evaluate_checkpoint", side_effect=fake_evaluate_checkpoint) as mock_evaluate:
                result = run_checkpoint_selection(
                    train_output_dir=train_output_dir,
                    dataset_path=Path(tmp_dir) / "dev.jsonl",
                    eval_cfg=eval_cfg,
                )

        self.assertEqual(mock_resolve.call_count, 1)
        self.assertEqual(mock_runner_cls.call_count, 1)
        self.assertEqual(mock_evaluate.call_count, 2)
        self.assertTrue(result["ranking_path"].endswith("dev_ranking.json"))
        self.assertTrue(result["best_path"].endswith("best_checkpoint.json"))


if __name__ == "__main__":
    unittest.main()
