from __future__ import annotations

from pathlib import Path

from post_train.data import prepare_benchmark_artifact
from post_train.io import write_jsonl

AIME_DATASETS: dict[int, dict[str, str]] = {
    24: {
        "dataset_name": "math-ai/aime24",
        "source": "aime24",
        "output": "data/eval/aime24_test.jsonl",
    },
    25: {
        "dataset_name": "math-ai/aime25",
        "source": "aime25",
        "output": "data/eval/aime25_test.jsonl",
    },
}


def resolve_aime_dataset(year: int) -> dict[str, str]:
    if year not in AIME_DATASETS:
        raise ValueError(f"暂不支持 AIME{year}，当前只支持 24 和 25。")
    return dict(AIME_DATASETS[year])


def prepare_aime_eval_artifact(
    *,
    year: int,
    split: str = "test",
    cache_dir: str | None = None,
    config_name: str | None = None,
    dataset_name: str | None = None,
    source: str | None = None,
) -> list[dict]:
    defaults = resolve_aime_dataset(year)
    return prepare_benchmark_artifact(
        dataset_name=dataset_name or defaults["dataset_name"],
        config_name=config_name,
        split=split,
        source=source or defaults["source"],
        cache_dir=cache_dir,
    )


def write_aime_eval_artifact(
    *,
    year: int,
    rows: list[dict],
    output_path: str | Path | None = None,
) -> Path:
    defaults = resolve_aime_dataset(year)
    return write_jsonl(output_path or defaults["output"], rows)
