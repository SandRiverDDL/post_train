from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from post_train.eval import build_model_output_path
from post_train.io import ensure_parent, read_jsonl, write_jsonl

DEFAULT_REGISTRY_PATH = Path("experiments/registry.jsonl")
DEFAULT_SUMMARY_MD_PATH = Path("experiments/summary.md")
DEFAULT_SUMMARY_CSV_PATH = Path("experiments/summary.csv")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_repo_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(candidate)


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def infer_route_from_config_path(config_path: str | Path) -> str:
    parts = Path(config_path).parts
    if "configs" in parts:
        index = parts.index("configs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def infer_parent_run_id(model_name: str | Path | None) -> str:
    if not model_name:
        return ""
    model_path = build_model_output_path(str(model_name))
    if not model_path.parts:
        return ""
    return str(model_path)


def _display_train_dataset(value: str | Path | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text or text == ".":
        return "unknown"
    return text


def _display_parent_model(value: str | Path | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    path = Path(text)
    parts = path.parts
    if "models--" in text and "snapshots" in parts:
        for part in parts:
            if part.startswith("models--"):
                return part.replace("models--", "").replace("--", "/")
    if "outputs" in parts:
        return infer_parent_run_id(text) or text
    return text


def _dev_summary_from_output_dir(output_dir: str | Path) -> dict[str, Any]:
    best_payload = _read_json_if_exists(Path(output_dir) / "dev_eval" / "best_checkpoint.json")
    if best_payload is None:
        return {
            "dataset": "",
            "selection_metric": "",
            "best_checkpoint_path": "",
            "best_global_step": None,
            "metrics": {},
            "ranking_path": "",
        }
    return {
        "dataset": str(best_payload.get("dev_dataset", "")),
        "selection_metric": str(best_payload.get("selection_metric", "")),
        "best_checkpoint_path": str(best_payload.get("checkpoint_path", "")),
        "best_global_step": best_payload.get("global_step"),
        "metrics": dict(best_payload.get("metrics", {})),
        "ranking_path": str(best_payload.get("ranking_path", "")),
    }


def _collect_benchmark_results_for_model(model_name: str | Path, *, eval_root: str | Path = "outputs/eval") -> dict[str, Any]:
    model_output_dir = Path(eval_root) / build_model_output_path(str(model_name))
    if not model_output_dir.exists():
        return {}
    benchmarks: dict[str, Any] = {}
    for task_dir in sorted(path for path in model_output_dir.iterdir() if path.is_dir()):
        result_path = task_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        benchmarks[task_dir.name] = {
            "result_path": _relative_repo_path(result_path),
            "metrics": dict(payload.get("metrics", {})),
        }
    return benchmarks


def write_run_summary(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    summary_name: str = "run_summary.json",
) -> Path:
    summary_path = ensure_parent(Path(output_dir) / summary_name)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def append_run_registry(summary: dict[str, Any], *, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    registry_file = Path(registry_path)
    rows = read_jsonl(registry_file) if registry_file.exists() else []
    rows = [row for row in rows if row.get("run_id") != summary.get("run_id")]
    rows.append(summary)
    return write_jsonl(registry_file, rows)


def build_training_run_summary(
    *,
    route: str,
    config_path: str | Path,
    output_dir: str | Path,
    train_dataset: str | Path,
    base_model: str | Path,
    status: str = "finished",
    notes: str = "",
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    dev_summary = _dev_summary_from_output_dir(resolved_output_dir)
    benchmarks = _collect_benchmark_results_for_model(resolved_output_dir)
    return {
        "run_id": _relative_repo_path(resolved_output_dir),
        "route": route,
        "source": "registered",
        "status": status,
        "created_at": _utc_now_iso(),
        "config_path": _relative_repo_path(config_path),
        "output_dir": _relative_repo_path(resolved_output_dir),
        "train_dataset": _relative_repo_path(train_dataset),
        "base_model": str(base_model),
        "parent_run_id": infer_parent_run_id(base_model),
        "best_checkpoint_path": dev_summary["best_checkpoint_path"],
        "best_global_step": dev_summary["best_global_step"],
        "dev_dataset": dev_summary["dataset"],
        "dev_selection_metric": dev_summary["selection_metric"],
        "dev_metrics": dev_summary["metrics"],
        "benchmarks": benchmarks,
        "notes": notes,
    }


def build_on_policy_loop_run_summary(
    *,
    config_path: str | Path,
    loop_cfg,
    final_summary: dict[str, Any],
    status: str = "finished",
    notes: str = "",
) -> dict[str, Any]:
    best_model_path = str(final_summary.get("best_model_path", ""))
    benchmarks = _collect_benchmark_results_for_model(best_model_path) if best_model_path else {}
    return {
        "run_id": _relative_repo_path(loop_cfg.round_base_dir),
        "route": "on_policy",
        "source": "registered",
        "status": status,
        "created_at": _utc_now_iso(),
        "config_path": _relative_repo_path(config_path),
        "output_dir": _relative_repo_path(loop_cfg.round_base_dir),
        "train_dataset": "",
        "base_model": str(loop_cfg.seed_model),
        "parent_run_id": infer_parent_run_id(loop_cfg.seed_model),
        "best_checkpoint_path": best_model_path,
        "best_global_step": None,
        "dev_dataset": str(final_summary.get("stop_dataset", "")),
        "dev_selection_metric": "normalized_accuracy",
        "dev_metrics": (
            {"normalized_accuracy": float(final_summary["best_holdout_accuracy"])}
            if final_summary.get("best_holdout_accuracy") is not None
            else {}
        ),
        "benchmarks": benchmarks,
        "notes": notes,
        "loop": {
            "completed_rounds": final_summary.get("completed_rounds"),
            "best_round_index": final_summary.get("best_round_index"),
            "stop_reason": final_summary.get("stop_reason", ""),
        },
    }


def register_training_run(
    *,
    route: str,
    config_path: str | Path,
    output_dir: str | Path,
    train_dataset: str | Path,
    base_model: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    summary = build_training_run_summary(
        route=route,
        config_path=config_path,
        output_dir=output_dir,
        train_dataset=train_dataset,
        base_model=base_model,
    )
    summary_path = write_run_summary(output_dir, summary)
    summary["summary_path"] = _relative_repo_path(summary_path)
    append_run_registry(summary, registry_path=registry_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def register_on_policy_loop_run(
    *,
    config_path: str | Path,
    loop_cfg,
    final_summary: dict[str, Any],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    summary = build_on_policy_loop_run_summary(
        config_path=config_path,
        loop_cfg=loop_cfg,
        final_summary=final_summary,
    )
    summary_path = write_run_summary(loop_cfg.round_base_dir, summary)
    summary["summary_path"] = _relative_repo_path(summary_path)
    append_run_registry(summary, registry_path=registry_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _discover_training_output_summary(output_dir: Path) -> dict[str, Any] | None:
    if not (output_dir / "adapter_config.json").exists():
        return None
    route = "unknown"
    if output_dir.name.startswith("stage1_"):
        route = "stage1"
    elif output_dir.name.startswith("stage2_"):
        route = "stage2"
    elif output_dir.name.startswith("simpo"):
        route = "simpo"
    elif output_dir.name.startswith("on_policy"):
        route = "on_policy"
    adapter_cfg = _read_json_if_exists(output_dir / "adapter_config.json") or {}
    base_model = adapter_cfg.get("base_model_name_or_path", "")
    summary = build_training_run_summary(
        route=route,
        config_path="",
        output_dir=output_dir,
        train_dataset="",
        base_model=base_model,
    )
    summary["discovered"] = True
    summary["source"] = "discovered"
    return summary


def _discover_loop_output_summary(output_dir: Path) -> dict[str, Any] | None:
    final_summary = _read_json_if_exists(output_dir / "final_summary.json")
    if final_summary is None:
        return None
    history = _read_json_if_exists(output_dir / "history.json") or {}
    summary = {
        "run_id": _relative_repo_path(output_dir),
        "route": "on_policy",
        "status": "finished",
        "created_at": _utc_now_iso(),
        "config_path": "",
        "output_dir": _relative_repo_path(output_dir),
        "train_dataset": "",
        "base_model": str(history.get("seed_model", "")),
        "parent_run_id": infer_parent_run_id(history.get("seed_model")),
        "best_checkpoint_path": str(final_summary.get("best_model_path", "")),
        "best_global_step": None,
        "dev_dataset": str(history.get("stop_dataset", "")),
        "dev_selection_metric": "normalized_accuracy",
        "dev_metrics": (
            {"normalized_accuracy": float(final_summary["best_holdout_accuracy"])}
            if final_summary.get("best_holdout_accuracy") is not None
            else {}
        ),
        "benchmarks": (
            _collect_benchmark_results_for_model(final_summary.get("best_model_path", ""))
            if final_summary.get("best_model_path")
            else {}
        ),
        "notes": "",
        "loop": {
            "completed_rounds": final_summary.get("completed_rounds"),
            "best_round_index": final_summary.get("best_round_index"),
            "stop_reason": final_summary.get("stop_reason", ""),
        },
        "discovered": True,
        "source": "discovered",
    }
    return summary


def discover_run_summaries(
    *,
    output_root: str | Path = "outputs",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    registry_file = Path(registry_path)
    if registry_file.exists():
        for row in read_jsonl(registry_file):
            discovered[str(row.get("output_dir", row.get("run_id", "")))] = row

    for output_dir in sorted(path for path in Path(output_root).iterdir() if path.is_dir() and path.name != "eval"):
        summary_path = output_dir / "run_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload.setdefault("source", "run_summary")
            discovered[str(payload.get("output_dir", output_dir))] = payload
            continue
        payload = _discover_loop_output_summary(output_dir)
        if payload is None:
            payload = _discover_training_output_summary(output_dir)
        if payload is not None:
            discovered[str(payload.get("output_dir", output_dir))] = payload

    return sorted(discovered.values(), key=lambda item: (str(item.get("route", "")), str(item.get("run_id", ""))))


def _metric_for_display(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _rank_map(rows: list[dict[str, Any]], benchmark_name: str) -> dict[str, int]:
    scored: list[tuple[str, float]] = []
    for row in rows:
        benchmark = row.get("benchmarks", {}).get(benchmark_name, {})
        score = _metric_value(benchmark.get("metrics", {}), "pass_at_1")
        if score is None:
            continue
        scored.append((str(row.get("run_id", "")), score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return {run_id: index + 1 for index, (run_id, _) in enumerate(scored)}


def _decorate_rows_for_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    math500_ranks = _rank_map(rows, "math500")
    gsm8k_ranks = _rank_map(rows, "gsm8k")
    for row in rows:
        item = dict(row)
        run_id = str(item.get("run_id", ""))
        math500_rank = math500_ranks.get(run_id)
        gsm8k_rank = gsm8k_ranks.get(run_id)
        item["math500_rank"] = math500_rank
        item["gsm8k_rank"] = gsm8k_rank
        item["complete_benchmarks"] = math500_rank is not None and gsm8k_rank is not None
        item["avg_rank"] = (
            (float(math500_rank) + float(gsm8k_rank)) / 2.0
            if item["complete_benchmarks"]
            else None
        )
        decorated.append(item)
    complete_rows = [row for row in decorated if row["complete_benchmarks"]]
    incomplete_rows = [row for row in decorated if not row["complete_benchmarks"]]
    complete_rows.sort(
        key=lambda row: (
            float(row["avg_rank"]),
            int(row["math500_rank"]),
            int(row["gsm8k_rank"]),
            -float(_metric_value(row.get("dev_metrics", {}), "normalized_accuracy") or 0.0),
            str(row.get("run_id", "")),
        )
    )
    incomplete_rows.sort(
        key=lambda row: (
            -float(_metric_value(row.get("benchmarks", {}).get("math500", {}).get("metrics", {}), "pass_at_1") or -1.0),
            -float(_metric_value(row.get("benchmarks", {}).get("gsm8k", {}).get("metrics", {}), "pass_at_1") or -1.0),
            -float(_metric_value(row.get("dev_metrics", {}), "normalized_accuracy") or 0.0),
            str(row.get("run_id", "")),
        )
    )
    return complete_rows + incomplete_rows


def write_experiment_summary_table(
    *,
    rows: list[dict[str, Any]],
    markdown_path: str | Path = DEFAULT_SUMMARY_MD_PATH,
    csv_path: str | Path = DEFAULT_SUMMARY_CSV_PATH,
) -> tuple[Path, Path]:
    ranked_rows = _decorate_rows_for_summary(rows)
    csv_rows: list[dict[str, str]] = []
    markdown_lines = [
        "# Experiment Summary",
        "",
        "## Complete Runs",
        "",
        "| Route | Run ID | Source | Train Dataset | Parent | Dev | math500 | gsm8k | math500 Rank | gsm8k Rank | Avg Rank |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    emitted_incomplete_header = False
    for row in ranked_rows:
        benchmarks = row.get("benchmarks", {})
        math500 = _metric_for_display(benchmarks.get("math500", {}).get("metrics", {}), "pass_at_1")
        gsm8k = _metric_for_display(benchmarks.get("gsm8k", {}).get("metrics", {}), "pass_at_1")
        dev = _metric_for_display(row.get("dev_metrics", {}), "normalized_accuracy")
        math500_rank = str(row.get("math500_rank") or "")
        gsm8k_rank = str(row.get("gsm8k_rank") or "")
        avg_rank = f"{row['avg_rank']:.2f}" if isinstance(row.get("avg_rank"), float) else ""
        csv_row = {
            "route": str(row.get("route", "")),
            "run_id": str(row.get("run_id", "")),
            "source": str(row.get("source", "unknown")),
            "train_dataset": _display_train_dataset(row.get("train_dataset", "")),
            "parent_run_id": _display_parent_model(row.get("parent_run_id", row.get("base_model", ""))),
            "dev_normalized_accuracy": dev,
            "math500_pass_at_1": math500,
            "gsm8k_pass_at_1": gsm8k,
            "math500_rank": math500_rank,
            "gsm8k_rank": gsm8k_rank,
            "avg_rank": avg_rank,
            "output_dir": str(row.get("output_dir", "")),
            "complete_benchmarks": "yes" if row.get("complete_benchmarks") else "no",
        }
        csv_rows.append(csv_row)
        if not row.get("complete_benchmarks") and not emitted_incomplete_header:
            markdown_lines.extend(
                [
                    "",
                    "## Incomplete Runs",
                    "",
                    "| Route | Run ID | Source | Train Dataset | Parent | Dev | math500 | gsm8k | math500 Rank | gsm8k Rank | Avg Rank |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            emitted_incomplete_header = True
        markdown_lines.append(
            "| {route} | {run_id} | {source} | {train_dataset} | {parent_run_id} | {dev_normalized_accuracy} | {math500_pass_at_1} | {gsm8k_pass_at_1} | {math500_rank} | {gsm8k_rank} | {avg_rank} |".format(
                **csv_row
            )
        )

    markdown_output = ensure_parent(markdown_path)
    markdown_output.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    csv_output = ensure_parent(csv_path)
    with csv_output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "route",
                "run_id",
                "source",
                "train_dataset",
                "parent_run_id",
                "dev_normalized_accuracy",
                "math500_pass_at_1",
                "gsm8k_pass_at_1",
                "math500_rank",
                "gsm8k_rank",
                "avg_rank",
                "complete_benchmarks",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    return markdown_output, csv_output
