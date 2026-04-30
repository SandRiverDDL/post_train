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
from post_train.rollout.math_sft import (
    build_math_query_pool,
    build_rejection_sampled_sft_rows,
    prepare_math_rollout_sft_dataset,
)


class MathRolloutSFTTest(unittest.TestCase):
    def test_build_math_query_pool_filters_levels_and_samples(self) -> None:
        raw_rows = [
            {"problem": "q1", "solution": "\\boxed{1}", "level": "Level 1", "type": "Algebra"},
            {"problem": "q2", "solution": "\\boxed{2}", "level": "Level 5", "type": "Geometry"},
            {"problem": "q3", "solution": "\\boxed{3}", "level": 5, "type": "Algebra"},
        ]
        with patch("post_train.rollout.math_sft.load_dataset_rows", return_value=raw_rows):
            rows, report = build_math_query_pool(dataset_name="toy/math", sample_size=2, seed=7, levels=[5])

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["meta"]["level"] for row in rows}, {5})
        self.assertEqual(report["filtered_level"], 1)
        self.assertEqual(report["selected_level_distribution"], {5: 2})

    def test_build_math_query_pool_expands_eleutherai_configs(self) -> None:
        def fake_load_dataset_rows(dataset_name, *, split, config_name=None, cache_dir=None):
            return [
                {"problem": f"{config_name}-q", "solution": "\\boxed{1}", "level": "Level 5", "type": "Algebra"},
            ]

        with patch("post_train.rollout.math_sft.load_dataset_rows", side_effect=fake_load_dataset_rows) as mock_load:
            rows, report = build_math_query_pool(
                dataset_name="EleutherAI/hendrycks_math",
                sample_size=7,
                seed=7,
            )

        self.assertEqual(mock_load.call_count, 7)
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(report["config_names"]), 7)
        self.assertTrue(all(row["meta"]["source_config"] for row in rows))

    def test_build_rejection_sampled_sft_rows_keeps_correct_only(self) -> None:
        raw_samples = [
            {
                "id": "q1",
                "question": "q",
                "final_answer": "1",
                "meta": {"level": 5},
                "responses": [
                    {"rollout_index": 0, "text": "bad", "boxed": False, "parse_ok": False, "correct": False, "output_tokens": 1},
                    {"rollout_index": 1, "text": "\\boxed{1}", "boxed": True, "parse_ok": True, "correct": True, "output_tokens": 2},
                ],
            }
        ]

        rows, report = build_rejection_sampled_sft_rows(raw_samples, target_count=1, seed=1, generation_model="model")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["solution"], "\\boxed{1}")
        self.assertEqual(rows[0]["meta"]["selector_name"], "boxed_parse_correct")
        self.assertEqual(report["candidate_count"], 1)

    def test_prepare_math_rollout_sft_dataset_writes_outputs(self) -> None:
        class FakeCandidate:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeOutput:
            def __init__(self, texts: list[str]) -> None:
                self.outputs = [FakeCandidate(text) for text in texts]

        class FakeLLM:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def generate(self, prompts, *, sampling_params, use_tqdm):
                return [FakeOutput(["\\boxed{1}", "wrong"]) for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        raw_rows = [
            {"problem": "q1", "solution": "\\boxed{1}", "level": "Level 5", "type": "Algebra"},
            {"problem": "q2", "solution": "\\boxed{1}", "level": "Level 5", "type": "Geometry"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mix_long_path = tmp_path / "mix.jsonl"
            mix_long_path.write_text(
                '{"id":"m1","question":"mq1","solution":"s1","final_answer":"1","meta":{}}\n'
                '{"id":"m2","question":"mq2","solution":"s2","final_answer":"2","meta":{}}\n',
                encoding="utf-8",
            )
            with patch("post_train.rollout.math_sft.load_dataset_rows", return_value=raw_rows), patch.dict(
                sys.modules,
                {"vllm": fake_vllm},
            ):
                report = prepare_math_rollout_sft_dataset(
                    model="fake/model",
                    output_dir=tmp_path / "out",
                    sample_size=2,
                    responses_per_prompt=2,
                    retained_count=2,
                    mix_long_sample_size=1,
                    mix_long_path=mix_long_path,
                    seed=3,
                )

            combined = read_jsonl(report["outputs"]["combined_train"])
            retained = read_jsonl(report["outputs"]["rollout_rejection_train"])

        self.assertEqual(len(retained), 2)
        self.assertEqual(len(combined), 3)
        self.assertEqual(report["combined"]["mix_long_count"], 1)


if __name__ == "__main__":
    unittest.main()
