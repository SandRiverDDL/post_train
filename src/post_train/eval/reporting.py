from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from post_train.io import ensure_parent, write_jsonl


TASK_ORDER = ["math500", "gsm8k", "aime24", "aime25", "math500_dev200"]


@dataclass(frozen=True)
class EvalReportPaths:
    metadata: Path
    registry: Path
    leaderboard: Path


def _metric_accuracy(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("pass_at_1", metrics.get("normalized_accuracy"))
    return float(value) if value is not None else None


def _task_from_dataset(dataset: str) -> str:
    stem = Path(dataset).stem
    if stem.endswith("_test"):
        return stem[: -len("_test")]
    if stem == "global_dev_math500_150":
        return "math500_dev150"
    return stem


def _task_from_result_path(path: Path, metrics: dict[str, Any]) -> str:
    task = path.parent.name
    if task and task != "outputs":
        return task
    return _task_from_dataset(str(metrics.get("dataset", "")))


def _infer_method(model: str) -> str:
    lowered = model.lower()
    if "grpo" in lowered:
        return "GRPO"
    if "lightning_opd" in lowered:
        return "Lightning-OPD"
    if "on_policy" in lowered or "opsft" in lowered:
        return "OPSFT"
    if "simpo" in lowered or "sipo" in lowered:
        return "SIMPO"
    if "stage1" in lowered or "sft" in lowered:
        return "SFT"
    if "qwen" in lowered:
        return "Base"
    return "Unknown"


def _default_label(model: str) -> str:
    return model.removeprefix("outputs/")


def _default_ignored(model: str) -> bool:
    lowered = model.lower()
    return "simpo" in lowered or "sipo" in lowered or "candidate_ddp_gpu1_6_b1" in lowered


def load_eval_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {"models": {}, "ignore_patterns": [], "manual_result_files": []}
    cfg = OmegaConf.load(metadata_path)
    return OmegaConf.to_container(cfg, resolve=True) or {}


def _read_result_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "source": "result_json",
        "result_path": str(path),
        "model": str(metrics.get("model", "")),
        "dataset": str(metrics.get("dataset", "")),
        "task": _task_from_result_path(path, metrics),
        "samples": metrics.get("samples"),
        "pass_at_1": _metric_accuracy(metrics),
        "stderr": metrics.get("pass_at_1_stderr", metrics.get("normalized_accuracy_stderr")),
        "boxed_rate": metrics.get("boxed_rate"),
        "parse_success_rate": metrics.get("parse_success_rate"),
        "avg_output_tokens": metrics.get("avg_output_tokens"),
    }


def _json_objects_from_markdown(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        brace_index = text.find("{", index)
        if brace_index < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[brace_index:])
        except json.JSONDecodeError:
            index = brace_index + 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        index = brace_index + end
    return rows


def _manual_row_from_object(obj: dict[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else obj
    model = obj.get("model") or obj.get("checkpoint_path") or metrics.get("model")
    dataset = obj.get("dataset") or metrics.get("dataset")
    if not model or not dataset:
        return None
    return {
        "source": "manual_md",
        "result_path": str(source_path),
        "model": str(model),
        "dataset": str(dataset),
        "task": _task_from_dataset(str(dataset)),
        "samples": metrics.get("samples"),
        "pass_at_1": _metric_accuracy(metrics),
        "stderr": metrics.get("pass_at_1_stderr", metrics.get("normalized_accuracy_stderr")),
        "boxed_rate": metrics.get("boxed_rate"),
        "parse_success_rate": metrics.get("parse_success_rate"),
        "avg_output_tokens": metrics.get("avg_output_tokens"),
    }


def discover_eval_rows(
    *,
    eval_root: str | Path,
    manual_result_files: list[str | Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(eval_root).glob("**/result.json")):
        row = _read_result_json(path)
        if row is not None:
            rows.append(row)
    for manual_file in manual_result_files:
        path = Path(manual_file)
        if not path.exists():
            continue
        for obj in _json_objects_from_markdown(path):
            row = _manual_row_from_object(obj, source_path=path)
            if row is not None:
                rows.append(row)
    return _deduplicate_rows(rows)


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"result_json": 2, "manual_md": 1}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["task"]), str(row["dataset"]))
        old = selected.get(key)
        if old is None or priority.get(str(row["source"]), 0) > priority.get(str(old["source"]), 0):
            selected[key] = row
    return sorted(selected.values(), key=lambda item: (str(item["model"]), str(item["task"]), str(item["dataset"])))


def enrich_eval_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    models = metadata.get("models", {}) if isinstance(metadata.get("models"), dict) else {}
    ignore_patterns = [str(item) for item in metadata.get("ignore_patterns", []) or []]
    enriched: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        model_meta = models.get(model) if isinstance(models.get(model), dict) else {}
        tracked = bool(model_meta) and bool(model_meta.get("track", True))
        ignored = bool(model_meta.get("ignore", False)) or _default_ignored(model)
        ignored = ignored or any(pattern in model for pattern in ignore_patterns)
        enriched.append(
            {
                **row,
                "method": str(model_meta.get("method") or _infer_method(model)),
                "label": str(model_meta.get("label") or _default_label(model)),
                "role": str(model_meta.get("role") or ""),
                "notes": str(model_meta.get("notes") or ""),
                "tracked": tracked,
                "ignore": ignored,
            }
        )
    return enriched


def build_metadata_template(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    # metadata 是人工维护的展示白名单，不自动追加所有 checkpoint。
    models = dict(metadata.get("models", {}) or {})
    return {
        "manual_result_files": list(metadata.get("manual_result_files", []) or []),
        "ignore_patterns": list(metadata.get("ignore_patterns", []) or []),
        "models": dict(sorted(models.items())),
    }


def write_metadata_template(path: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(OmegaConf.to_yaml(OmegaConf.create(metadata), resolve=True), encoding="utf-8")
    return output_path


def _format_score(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _best_rows_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        model = str(row["model"])
        task = str(row["task"])
        current = grouped.setdefault(model, {}).get(task)
        if current is None or (row.get("pass_at_1") or -1.0) > (current.get("pass_at_1") or -1.0):
            grouped[model][task] = row
    return grouped


def _sort_model_group(item: tuple[str, dict[str, dict[str, Any]]]) -> tuple[float, float, str]:
    model, task_rows = item
    math_score = task_rows.get("math500", {}).get("pass_at_1")
    gsm_score = task_rows.get("gsm8k", {}).get("pass_at_1")
    return (-(math_score if math_score is not None else -1.0), -(gsm_score if gsm_score is not None else -1.0), model)


def render_leaderboard(rows: list[dict[str, Any]]) -> str:
    tracked_rows = [row for row in rows if row.get("tracked")]
    active = [row for row in tracked_rows if not row.get("ignore")]
    ignored = [row for row in tracked_rows if row.get("ignore")]
    untracked_count = len([row for row in rows if not row.get("tracked")])
    lines = [
        "# Eval Leaderboard",
        "",
        "本文件由 `scripts/report_eval_results.py` 自动生成；不要手动编辑表格。",
        "",
        f"未被 `docs/analysis/eval_metadata.yaml` 显式跟踪的结果只写入 registry，不进入本表；当前隐藏 {untracked_count} 行。",
        "",
        "## Main Results",
        "",
        "| method | label | math500 | gsm8k | aime24 | aime25 | notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    grouped = _best_rows_by_model(active)
    for model, task_rows in sorted(grouped.items(), key=_sort_model_group):
        if not any(task in task_rows for task in TASK_ORDER[:4]):
            continue
        first = next(iter(task_rows.values()))
        scores = [_format_score(task_rows.get(task, {}).get("pass_at_1")) for task in TASK_ORDER[:4]]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(first.get("method", "")),
                    str(first.get("label", model)),
                    *scores,
                    str(first.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Task Details", ""])
    for task in TASK_ORDER:
        task_rows = [row for row in active if row["task"] == task]
        if not task_rows:
            continue
        lines.extend(
            [
                f"### {task}",
                "",
                "| score | stderr | boxed | avg tokens | samples | method | label | result |",
                "|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in sorted(task_rows, key=lambda item: (-(item.get("pass_at_1") or -1.0), str(item["model"]))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_score(row.get("pass_at_1")),
                        _format_score(row.get("stderr")),
                        _format_score(row.get("boxed_rate")),
                        f"{float(row['avg_output_tokens']):.1f}" if row.get("avg_output_tokens") is not None else "-",
                        str(row.get("samples") or "-"),
                        str(row.get("method", "")),
                        str(row.get("label", row["model"])),
                        str(row.get("result_path", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    if ignored:
        lines.extend(["## Ignored / Archived", "", "| task | score | method | label | reason |", "|---|---:|---|---|---|"])
        for row in sorted(ignored, key=lambda item: (str(item["model"]), str(item["task"]))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["task"]),
                        _format_score(row.get("pass_at_1")),
                        str(row.get("method", "")),
                        str(row.get("label", row["model"])),
                        str(row.get("notes", "")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def write_eval_report(
    *,
    eval_root: str | Path,
    paths: EvalReportPaths,
    manual_result_files: list[str | Path] | None = None,
    update_metadata: bool = True,
) -> tuple[list[dict[str, Any]], EvalReportPaths]:
    metadata = load_eval_metadata(paths.metadata)
    configured_manual_files = list(metadata.get("manual_result_files", []) or [])
    resolved_manual_files = list(manual_result_files if manual_result_files is not None else configured_manual_files)
    rows = discover_eval_rows(eval_root=eval_root, manual_result_files=resolved_manual_files)
    if update_metadata:
        metadata = build_metadata_template(rows, {**metadata, "manual_result_files": resolved_manual_files})
        write_metadata_template(paths.metadata, metadata)
    enriched = enrich_eval_rows(rows, metadata)
    write_jsonl(paths.registry, enriched)
    ensure_parent(paths.leaderboard).write_text(render_leaderboard(enriched), encoding="utf-8")
    return enriched, paths
