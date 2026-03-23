from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_train_corpora


class MergeTrainCorporaTest(unittest.TestCase):
    def test_main_merges_by_priority_and_records_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_a = tmp / "a.jsonl"
            input_b = tmp / "b.jsonl"
            eval_path = tmp / "eval.jsonl"
            output_path = tmp / "merged.jsonl"
            summary_path = tmp / "merged_summary.json"
            conflicts_path = tmp / "merged_conflicts.jsonl"
            manifest_path = tmp / "manifest.yaml"

            input_a.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a1", "question": "What is 2+2?", "final_answer": "4", "source": "big_math"}),
                        json.dumps({"id": "a2", "question": "What is 5+5?", "final_answer": "10", "source": "big_math"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            input_b.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "b1", "question": "What is 2+2?", "final_answer": "5", "source": "gsm8k"}),
                        json.dumps({"id": "b2", "question": "What is 7+1?", "final_answer": "8", "source": "gsm8k"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps({"id": "e1", "question": "What is 5+5?", "final_answer": "10", "source": "gsm8k"}) + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "inputs": [
                            {"name": "big_math", "path": str(input_a), "priority": 20},
                            {"name": "gsm8k", "path": str(input_b), "priority": 10},
                        ],
                        "dedup_against": [str(eval_path)],
                        "merge_output": str(output_path),
                        "merge_summary_output": str(summary_path),
                        "conflicts_output": str(conflicts_path),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch.object(sys, "argv", ["merge_train_corpora.py", "--manifest", str(manifest_path)]):
                merge_train_corpora.main()

            merged_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in merged_rows], ["a1", "b2"])

            conflicts = [json.loads(line) for line in conflicts_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["question"], "What is 2+2?")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["merged_count"], 2)
            self.assertEqual(summary["dedup_against_hits"], 1)
            self.assertEqual(summary["conflict_count"], 1)


if __name__ == "__main__":
    unittest.main()
