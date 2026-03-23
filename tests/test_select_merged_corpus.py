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

import select_merged_corpus


class SelectMergedCorpusTest(unittest.TestCase):
    def test_main_selects_with_min_max_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pool_path = tmp / "pool.jsonl"
            final_path = tmp / "selected.jsonl"
            summary_path = tmp / "selected_summary.json"
            manifest_path = tmp / "manifest.yaml"

            rows = []
            for index in range(5):
                rows.append({"id": f"g{index}", "question": f"gsm {index}", "final_answer": str(index), "source": "gsm8k"})
            for index in range(4):
                rows.append({"id": f"b{index}", "question": f"big {index}", "final_answer": str(index), "source": "big_math"})
            pool_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "merge_output": str(pool_path),
                        "selection": {
                            "target_size": 6,
                            "seed": 42,
                            "fill_strategy": "weighted",
                            "per_source": {
                                "gsm8k": {"min_count": 2, "max_count": 4},
                                "big_math": {"min_count": 1, "max_count": 3},
                            },
                            "weights": {
                                "gsm8k": 1.0,
                                "big_math": 1.0,
                            },
                        },
                        "final_output": str(final_path),
                        "selection_summary_output": str(summary_path),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch.object(sys, "argv", ["select_merged_corpus.py", "--manifest", str(manifest_path)]):
                select_merged_corpus.main()

            selected_rows = [json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(selected_rows), 6)
            source_counts: dict[str, int] = {}
            for row in selected_rows:
                source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            self.assertEqual(source_counts, {"gsm8k": 3, "big_math": 3})

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_count"], 6)
            self.assertEqual(summary["source_stats"]["gsm8k"]["selected_count"], 3)
            self.assertEqual(summary["source_stats"]["big_math"]["selected_count"], 3)

    def test_main_fails_when_min_count_cannot_be_satisfied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pool_path = tmp / "pool.jsonl"
            manifest_path = tmp / "manifest.yaml"

            pool_path.write_text(
                json.dumps({"id": "g0", "question": "gsm 0", "final_answer": "0", "source": "gsm8k"}) + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "merge_output": str(pool_path),
                        "selection": {
                            "target_size": 2,
                            "seed": 42,
                            "fill_strategy": "weighted",
                            "per_source": {
                                "gsm8k": {"min_count": 2, "max_count": 2},
                            },
                        },
                        "final_output": str(tmp / "selected.jsonl"),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch.object(sys, "argv", ["select_merged_corpus.py", "--manifest", str(manifest_path)]):
                with self.assertRaisesRegex(ValueError, "min_count"):
                    select_merged_corpus.main()


if __name__ == "__main__":
    unittest.main()
