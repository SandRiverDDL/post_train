from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.io import read_jsonl
from post_train.rollout.teacher_sft import (
    build_teacher_sft_prompts,
    merge_teacher_sft_shards,
    run_teacher_sft_shard,
)


class TeacherSFTRolloutTest(unittest.TestCase):
    def test_build_prompts_adds_benchmark_suffix_when_prompt_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "candidate.jsonl"
            source.write_text(
                '{"id":"q1","question":"1+1?","final_answer":"2","meta":{"source":"toy"}}\n',
                encoding="utf-8",
            )

            report = build_teacher_sft_prompts(
                prompt_source=source,
                output_dir=tmp_path / "out",
                sample_size=1,
                seed=1,
            )
            rows = read_jsonl(report["outputs"]["prompts"])

        self.assertEqual(rows[0]["prompt"], "1+1?\nPlease reason step by step, and put your final answer within \\boxed{}.")
        self.assertEqual(rows[0]["meta"]["global_query_index"], 0)

    def test_run_shard_uses_qwen3_non_thinking_chat_template(self) -> None:
        class FakeTokenizer:
            chat_template = "fake"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
                self.kwargs = kwargs
                return f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

            def __call__(self, text, add_special_tokens=False):
                class Encoded:
                    input_ids = text.split()

                return Encoded()

        fake_tokenizer = FakeTokenizer()

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return fake_tokenizer

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer

        class FakeCandidate:
            text = "Therefore \\boxed{2}"
            finish_reason = "stop"

        class FakeOutput:
            outputs = [FakeCandidate()]

        class FakeLLM:
            def __init__(self, *args, **kwargs) -> None:
                self.kwargs = kwargs

            def generate(self, prompts, sampling_params, use_tqdm):
                self.prompts = prompts
                return [FakeOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompts = tmp_path / "out" / "prompts.jsonl"
            prompts.parent.mkdir(parents=True)
            prompts.write_text(
                '{"id":"q1","question":"1+1?","final_answer":"2","prompt":"1+1?\\nPlease reason step by step, and put your final answer within \\\\boxed{}.","meta":{"global_query_index":0}}\n',
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
                report = run_teacher_sft_shard(
                    output_dir=tmp_path / "out",
                    num_shards=1,
                    shard_index=0,
                    model="fake/model",
                    use_chat_template=True,
                    chat_template_enable_thinking=False,
                    max_new_tokens=16,
                )
            raw = read_jsonl(report["outputs"]["raw_rollouts"])

        self.assertFalse(fake_tokenizer.kwargs["enable_thinking"])
        self.assertIn("<think>\n\n</think>\n\n", raw[0]["rendered_prompt"])
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["truncated_count"], 0)

    def test_merge_reports_length_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shard = tmp_path / "out" / "shards" / "shard0-of-1"
            shard.mkdir(parents=True)
            (shard / "raw_rollouts.jsonl").write_text(
                '{"id":"q1","question":"1+1?","final_answer":"2","prompt":"p","rendered_prompt":"<think>\\n\\n</think>\\n\\n","responses":[{"rollout_index":0,"text":"\\\\boxed{2}","boxed":true,"parse_ok":true,"correct":true,"output_tokens":3,"output_tokens_whitespace":1,"finish_reason":"stop","truncated":false}],"meta":{"global_query_index":0,"shard_index":0}}\n',
                encoding="utf-8",
            )
            (shard / "report.json").write_text('{"response_count":1}\n', encoding="utf-8")

            report = merge_teacher_sft_shards(output_dir=tmp_path / "out", num_shards=1, model="fake/model")
            merged = read_jsonl(report["outputs"]["raw_rollouts"])

        self.assertEqual(len(merged), 1)
        self.assertEqual(report["analysis"]["correct_rate"], 1.0)
        self.assertEqual(report["analysis"]["trunc_rate"], 0.0)
        self.assertEqual(report["analysis"]["rendered_prompt_empty_think_prefix_count"], 1)


if __name__ == "__main__":
    unittest.main()
