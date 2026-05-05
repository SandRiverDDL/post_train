from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from post_train.verl_opd.data import build_math_level_dataset, build_mix_dataset, build_smoke_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_dapo(path: Path, count: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "prompt": f"DAPO problem {index}?",
            "solution": f"solution {index}",
            "data_source": "math_dapo",
            "source_prompt": [{"role": "user", "content": f"source prompt {index}"}],
            "ability": "math",
            "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": str(index)},
            "extra_info": {"index": f"dapo-{index}"},
        }
        for index in range(count)
    ]
    Dataset.from_list(rows).to_parquet(str(path))


def test_build_smoke_dataset_writes_verl_prompt_schema(tmp_path: Path) -> None:
    dapo_path = tmp_path / "dapo.parquet"
    output_dir = tmp_path / "smoke"
    _write_dapo(dapo_path)

    report = build_smoke_dataset(dapo_source=dapo_path, output_dir=output_dir, sample_size=3, seed=7)
    dataset = Dataset.from_parquet(report["outputs"]["train"])
    first = dataset[0]

    assert report["counts"]["opd_dapo"] == 3
    assert len(dataset) == 3
    assert first["prompt"][0]["role"] == "user"
    assert "\\boxed{}" in first["prompt"][0]["content"]
    assert first["data_source"] == "math_dapo"
    assert first["extra_info"]["source"] == "opd_dapo"


def test_build_mix_dataset_uses_candidate_to_dapo_ratio(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate_pool.jsonl"
    dapo_path = tmp_path / "dapo.parquet"
    output_dir = tmp_path / "mix"
    _write_jsonl(
        candidate_path,
        [
            {"id": "c0", "question": "candidate problem 0?", "final_answer": "0", "meta": {"source": "test"}},
            {"id": "c1", "question": "candidate problem 1?", "final_answer": "1", "meta": {"source": "test"}},
        ],
    )
    _write_dapo(dapo_path, count=10)

    report = build_mix_dataset(
        candidate_source=candidate_path,
        dapo_source=dapo_path,
        output_dir=output_dir,
        candidate_size=2,
        dapo_size=8,
        seed=11,
    )
    dataset = Dataset.from_parquet(report["outputs"]["train"])
    sources = [row["extra_info"]["source"] for row in dataset]

    assert len(dataset) == 10
    assert sources.count("opd_candidate") == 2
    assert sources.count("opd_dapo") == 8
    assert set(dataset["data_source"]) == {"math_dapo"}


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        assert kwargs["enable_thinking"] is False
        content = messages[0]["content"]
        length = 300 if "too long" in content else 30
        return list(range(length))


def test_build_math_level_dataset_filters_prompt_tokens_and_writes_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "math-levels"
    rows = [
        {
            "id": "l3-short",
            "question": "short level 3",
            "final_answer": "3",
            "meta": {"source_dataset": "toy/math", "source_split": "train", "source_index": 0, "source_config": "algebra", "level": 3, "type": "Algebra"},
        },
        {
            "id": "l4-short",
            "question": "short level 4",
            "final_answer": "4",
            "meta": {"source_dataset": "toy/math", "source_split": "train", "source_index": 1, "source_config": "geometry", "level": 4, "type": "Geometry"},
        },
        {
            "id": "l4-long",
            "question": "too long level 4",
            "final_answer": "5",
            "meta": {"source_dataset": "toy/math", "source_split": "train", "source_index": 2, "source_config": "geometry", "level": 4, "type": "Geometry"},
        },
    ]
    pool_report = {"eligible_rows": len(rows), "selected_level_distribution": {3: 1, 4: 2}}

    with (
        patch("post_train.verl_opd.data.build_math_query_pool", return_value=(rows, pool_report)),
        patch("post_train.verl_opd.data.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()),
    ):
        report = build_math_level_dataset(
            output_dir=output_dir,
            sample_size=2,
            seed=7,
            max_prompt_tokens=256,
            tokenizer_name="toy-tokenizer",
            cache_dir=None,
        )

    dataset = Dataset.from_parquet(report["outputs"]["train"])
    levels = {row["extra_info"]["level"] for row in dataset}
    prompt_lengths = [row["extra_info"]["prompt_token_length"] for row in dataset]

    assert report["counts"]["level_filtered"] == 3
    assert report["counts"]["filtered_missing_answer"] == 0
    assert report["counts"]["filtered_over_prompt_tokens"] == 1
    assert report["counts"]["eligible_after_prompt_filter"] == 2
    assert len(dataset) == 2
    assert levels == {3, 4}
    assert max(prompt_lengths) <= 256
    assert dataset[0]["prompt"][0]["role"] == "user"
    assert "\\boxed{}" in dataset[0]["prompt"][0]["content"]
    assert all(row["reward_model"]["ground_truth"] for row in dataset)
