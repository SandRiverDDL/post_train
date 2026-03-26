from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_simpo_config
from post_train.simpo import build_quantization_kwargs, build_training_args


class SimPOTrainTest(unittest.TestCase):
    def test_load_simpo_config_supports_new_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "simpo.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "base_model_name: /tmp/base",
                        "model_name_or_path: /tmp/adapter",
                        "train_dataset: /tmp/train.jsonl",
                        "output_dir: /tmp/output",
                        "load_in_4bit: true",
                        "save_strategy: steps",
                        "save_steps: 12",
                        "resume_from_checkpoint: /tmp/output/checkpoint-12",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_simpo_config(config_path)

        self.assertTrue(cfg.load_in_4bit)
        self.assertEqual(cfg.save_steps, 12)
        self.assertEqual(cfg.resume_from_checkpoint, "/tmp/output/checkpoint-12")

    def test_build_training_args_uses_resume_related_fields(self) -> None:
        cfg = load_simpo_config("configs/simpo.yaml")
        training_args = build_training_args(cfg)

        self.assertEqual(training_args.logging_steps, 5)
        self.assertEqual(str(training_args.save_strategy), "SaveStrategy.STEPS")
        self.assertEqual(training_args.save_steps, 25)
        self.assertEqual(training_args.max_prompt_length, 256)
        self.assertEqual(training_args.max_length, 768)

    def test_build_quantization_kwargs_uses_4bit_defaults(self) -> None:
        cfg = load_simpo_config("configs/simpo.yaml")

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_bf16_supported", return_value=True
        ):
            kwargs = build_quantization_kwargs(cfg)

        self.assertIn("quantization_config", kwargs)
        quant_config = kwargs["quantization_config"]
        self.assertTrue(quant_config.load_in_4bit)
        self.assertEqual(quant_config.bnb_4bit_quant_type, "nf4")
        self.assertEqual(kwargs["device_map"], "auto")


if __name__ == "__main__":
    unittest.main()
