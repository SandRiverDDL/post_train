from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.on_policy_query_strategy import build_query_sampler_state
from post_train.on_policy_trainset import build_mixed_plus_anchor_dataset


class OnPolicyTrainsetTest(unittest.TestCase):
    def test_build_mixed_plus_anchor_dataset_uses_ratio_and_excludes_same_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mixed_path = tmp_path / "mixed.jsonl"
            mixed_path.write_text(
                "\n".join(
                    [
                        '{"id":"m1","question":"Q1","solution":"S1","final_answer":"1","meta":{}}',
                        '{"id":"m2","question":"Q2","solution":"S2","final_answer":"2","meta":{}}',
                        '{"id":"m3","question":"Q3","solution":"S3","final_answer":"3","meta":{}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            anchor_rows = [
                {"id": "m2", "question": "Q2", "solution": "A-dup-id", "final_answer": "2", "meta": {}},
                {"id": "a1", "question": "Q3", "solution": "A-dup-question", "final_answer": "3", "meta": {}},
                {"id": "a2", "question": "Q4", "solution": "A2", "final_answer": "4", "meta": {}},
                {"id": "a3", "question": "Q5", "solution": "A3", "final_answer": "5", "meta": {}},
            ]
            sampler_state = build_query_sampler_state(anchor_rows, seed=42)

            output_path, report = build_mixed_plus_anchor_dataset(
                mixed_retained_path=mixed_path,
                anchor_rows=anchor_rows,
                anchor_sampler_state=sampler_state,
                anchor_share=0.25,
                output_path=tmp_path / "train.jsonl",
            )

            rows = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(report["mixed_count"], 3)
        self.assertEqual(report["anchor_target"], 1)
        self.assertEqual(report["anchor_count"], 1)
        self.assertEqual(report["train_sample_count"], 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(any('"id": "a2"' in row or '"id": "a3"' in row for row in rows))


if __name__ == "__main__":
    unittest.main()
