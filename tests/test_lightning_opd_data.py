from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.io import read_jsonl, write_jsonl
from post_train.lightning_opd.data import (
    _sampled_logprobs_and_topk,
    build_lightning_opd_prompts,
    merge_lightning_opd_shards,
    select_shard_rows,
)


class LightningOPDDataTest(unittest.TestCase):
    def test_build_prompts_samples_candidate_pool_reproducibly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "candidate.jsonl"
            write_jsonl(
                source,
                [
                    {"id": str(index), "question": f"q{index}", "final_answer": str(index), "meta": {"source": "candidate"}}
                    for index in range(5)
                ],
            )

            report = build_lightning_opd_prompts(
                prompt_source=source,
                output_dir=tmp_path / "out",
                sample_size=3,
                seed=7,
            )
            rows = read_jsonl(report["outputs"]["prompts"])

        self.assertEqual(len(rows), 3)
        self.assertTrue(all("prompt" in row for row in rows))
        self.assertEqual([row["id"] for row in rows], ["2", "1", "3"])

    def test_select_shard_rows_uses_modulo_split(self) -> None:
        rows = [{"id": str(index)} for index in range(6)]

        self.assertEqual([row["id"] for row in select_shard_rows(rows, num_shards=2, shard_index=0)], ["0", "2", "4"])
        self.assertEqual([row["id"] for row in select_shard_rows(rows, num_shards=2, shard_index=1)], ["1", "3", "5"])

    def test_merge_lightning_opd_shards_writes_train_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "out"
            for shard_index in [0, 1]:
                shard_dir = base / "shards" / f"shard{shard_index}-of-2"
                shard_dir.mkdir(parents=True)
                write_jsonl(
                    shard_dir / "raw_rollouts.jsonl",
                    [{"id": f"q{shard_index}", "response": "\\boxed{1}"}],
                )
                write_jsonl(
                    shard_dir / "teacher_topk.jsonl",
                    [
                        {
                            "id": f"q{shard_index}",
                            "question": "q",
                            "final_answer": "1",
                            "prompt": "p",
                            "response": "\\boxed{1}",
                            "input_ids": [10, 11, 12],
                            "response_mask": [0, 1, 1],
                            "teacher_token_logprobs": [-0.1, -0.2],
                            "teacher_topk_token_ids": [[11, 1], [12, 2]],
                            "teacher_topk_logprobs": [[-0.1, -1.0], [-0.2, -1.2]],
                            "meta": {"global_query_index": shard_index, "response_tokens": 2},
                        }
                    ],
                )

            report = merge_lightning_opd_shards(output_dir=base, num_shards=2)
            train_rows = read_jsonl(report["outputs"]["train"])
            report_payload = json.loads((base / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(len(train_rows), 2)
        self.assertEqual(report["shape_errors"], 0)
        self.assertEqual(report["top_k_values"], [2])
        self.assertEqual(report_payload["train_rows"], 2)

    def test_sampled_logprobs_matches_dense_log_softmax(self) -> None:
        import torch

        logits = torch.tensor([[1.0, 2.0, 4.0], [0.5, -1.0, 0.0]], dtype=torch.float32)
        target_ids = torch.tensor([2, 0], dtype=torch.long)

        sampled_logprobs, topk_ids, topk_logprobs = _sampled_logprobs_and_topk(logits, target_ids, top_k=2)
        dense_logprobs = torch.log_softmax(logits, dim=-1)

        self.assertEqual(topk_ids, [[2, 1], [0, 2]])
        self.assertTrue(torch.allclose(torch.tensor(sampled_logprobs), dense_logprobs[torch.arange(2), target_ids]))
        self.assertTrue(torch.allclose(torch.tensor(topk_logprobs[0]), dense_logprobs[0, torch.tensor(topk_ids[0])]))


if __name__ == "__main__":
    unittest.main()
