#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.io import ensure_parent, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 merge 后的 pool 中按预算选择最终训练集。")
    parser.add_argument("--manifest", required=True, help="manifest YAML 路径")
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _group_rows_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        grouped.setdefault(source, []).append(row)
    return grouped


def _allocate_required(groups: dict[str, list[dict]], per_source: dict[str, dict]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    allocated: dict[str, int] = {}
    stats: dict[str, dict[str, int]] = {}
    for source, rows in groups.items():
        config = per_source.get(source, {})
        min_count = int(config.get("min_count", 0))
        max_count = int(config.get("max_count", len(rows)))
        available = len(rows)
        if min_count > available:
            raise ValueError(f"source={source} 无法满足 min_count：{available} < {min_count}")
        effective_max = min(max_count, available)
        if min_count > effective_max:
            raise ValueError(f"source={source} 的 min_count 大于 max_count。")
        allocated[source] = min_count
        stats[source] = {
            "available_count": available,
            "min_count": min_count,
            "max_count": effective_max,
            "selected_count": min_count,
        }
    return allocated, stats


def _fill_remaining_capacity(
    allocated: dict[str, int],
    stats: dict[str, dict[str, int]],
    *,
    weights: dict[str, float],
    remaining_slots: int,
) -> None:
    while remaining_slots > 0:
        candidates = [
            source
            for source, source_stats in stats.items()
            if allocated[source] < source_stats["max_count"]
        ]
        if not candidates:
            break
        candidates.sort(key=lambda source: (-float(weights.get(source, 1.0)), source))
        assigned = False
        for source in candidates:
            if remaining_slots <= 0:
                break
            if allocated[source] >= stats[source]["max_count"]:
                continue
            allocated[source] += 1
            stats[source]["selected_count"] += 1
            remaining_slots -= 1
            assigned = True
        if not assigned:
            break


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    selection = dict(manifest.get("selection") or {})
    pool_path = Path(manifest.get("merge_output") or manifest.get("output") or "")
    if not pool_path:
        raise ValueError("manifest 必须提供 merge_output。")
    rows = read_jsonl(pool_path)
    grouped = _group_rows_by_source(rows)

    target_size = int(selection["target_size"])
    seed = int(selection.get("seed", 42))
    fill_strategy = str(selection.get("fill_strategy", "weighted"))
    if fill_strategy != "weighted":
        raise ValueError(f"不支持的 fill_strategy: {fill_strategy}")
    per_source = dict(selection.get("per_source") or {})
    weights = {str(key): float(value) for key, value in dict(selection.get("weights") or {}).items()}

    allocated, stats = _allocate_required(grouped, per_source)
    required_total = sum(allocated.values())
    if required_total > target_size:
        raise ValueError(f"min_count 总和超过 target_size：{required_total} > {target_size}")

    remaining_slots = target_size - required_total
    _fill_remaining_capacity(allocated, stats, weights=weights, remaining_slots=remaining_slots)
    selected_total = sum(allocated.values())
    if selected_total < target_size:
        raise ValueError(f"在 max_count 约束下样本不足：{selected_total} < {target_size}")

    rng = random.Random(seed)
    selected_rows: list[dict] = []
    for source in sorted(grouped):
        source_rows = list(grouped[source])
        rng.shuffle(source_rows)
        selected_rows.extend(source_rows[: allocated[source]])
    rng.shuffle(selected_rows)

    final_output = ensure_parent(manifest["final_output"])
    summary_output = ensure_parent(
        manifest.get("selection_summary_output")
        or final_output.with_name(f"{final_output.stem}_summary.json")
    )
    write_jsonl(final_output, selected_rows)
    with summary_output.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "pool_count": len(rows),
                "target_size": target_size,
                "selected_count": len(selected_rows),
                "fill_strategy": fill_strategy,
                "source_stats": stats,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(selected_rows)} rows to {final_output}")
    print(f"Wrote selection summary to {summary_output}")


if __name__ == "__main__":
    main()
