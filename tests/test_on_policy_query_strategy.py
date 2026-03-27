from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.on_policy_query_strategy import (
    MIXED_SOURCE_CANDIDATE,
    build_mixed_query_strategy,
    sample_mixed_round_queries,
    update_mixed_query_strategy,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class OnPolicyQueryStrategyTest(unittest.TestCase):
    def test_sample_mixed_round_queries_respects_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bootstrap_path = tmp_path / "bootstrap_raw_samples.jsonl"
            candidate_path = tmp_path / "candidate_pool.jsonl"
            _write_jsonl(
                bootstrap_path,
                [
                    {
                        "id": f"b{i}",
                        "question": f"Bootstrap {i}",
                        "final_answer": str(i),
                        "meta": {"source": "boot"},
                        "response_summary": {"response_count": 4, "correct_count": 4},
                    }
                    for i in range(4)
                ],
            )
            _write_jsonl(
                candidate_path,
                [
                    {
                        "id": f"c{i}",
                        "question": f"Candidate {i}",
                        "final_answer": str(i),
                        "meta": {"source": "cand"},
                    }
                    for i in range(6)
                ],
            )

            state = build_mixed_query_strategy(
                bootstrap_all_correct_raw_samples=bootstrap_path,
                candidate_query_file=candidate_path,
                data_base_dir=tmp_path / "data",
                seed=42,
                bootstrap_ratio=0.3,
                candidate_ratio=0.7,
                candidate_freeze_all_correct_hits=2,
            )
            round_rows, report = sample_mixed_round_queries(state, requested_count=10)

        self.assertEqual(len(round_rows), 10)
        self.assertEqual(report["source_mix"]["bootstrap_target"], 3)
        self.assertEqual(report["source_mix"]["candidate_target"], 7)
        self.assertEqual(report["source_mix"]["bootstrap_sampled"], 4)
        self.assertEqual(report["source_mix"]["candidate_sampled"], 6)
        self.assertEqual(report["source_mix"]["quota_shortage"], 0)

    def test_update_mixed_query_strategy_freezes_candidates_after_two_all_correct_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bootstrap_path = tmp_path / "bootstrap_raw_samples.jsonl"
            candidate_path = tmp_path / "candidate_pool.jsonl"
            _write_jsonl(bootstrap_path, [])
            _write_jsonl(
                candidate_path,
                [
                    {
                        "id": "c1",
                        "question": "Candidate 1",
                        "final_answer": "1",
                        "meta": {"source": "cand"},
                    },
                    {
                        "id": "c2",
                        "question": "Candidate 2",
                        "final_answer": "2",
                        "meta": {"source": "cand"},
                    },
                ],
            )

            state = build_mixed_query_strategy(
                bootstrap_all_correct_raw_samples=bootstrap_path,
                candidate_query_file=candidate_path,
                data_base_dir=tmp_path / "data",
                seed=7,
                bootstrap_ratio=0.0,
                candidate_ratio=1.0,
                candidate_freeze_all_correct_hits=2,
            )

            round1_rows, _ = sample_mixed_round_queries(state, requested_count=2)
            raw_samples_round1 = tmp_path / "round1_raw.jsonl"
            _write_jsonl(
                raw_samples_round1,
                [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "final_answer": row["final_answer"],
                        "meta": row["meta"],
                        "response_summary": {"response_count": 4, "correct_count": 4},
                    }
                    for row in round1_rows
                ],
            )
            update_mixed_query_strategy(state, raw_samples_path=raw_samples_round1, round_index=1)

            round2_rows, _ = sample_mixed_round_queries(state, requested_count=2)
            raw_samples_round2 = tmp_path / "round2_raw.jsonl"
            _write_jsonl(
                raw_samples_round2,
                [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "final_answer": row["final_answer"],
                        "meta": row["meta"],
                        "response_summary": {"response_count": 4, "correct_count": 4},
                    }
                    for row in round2_rows
                ],
            )
            report = update_mixed_query_strategy(state, raw_samples_path=raw_samples_round2, round_index=2)

        self.assertEqual({row["meta"]["on_policy_query_strategy_source"] for row in round1_rows}, {MIXED_SOURCE_CANDIDATE})
        self.assertEqual(report["candidate_newly_frozen"], 2)
        self.assertEqual(report["candidate_frozen_count"], 2)


if __name__ == "__main__":
    unittest.main()
