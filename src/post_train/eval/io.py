from __future__ import annotations

import json
import re
from inspect import getsource
from pathlib import Path
from typing import Any

from post_train.config import EvalTaskConfig
from post_train.io import ensure_parent


def _sanitize_output_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "unknown"


def build_model_output_path(model_name: str) -> Path:
    model_path = Path(model_name)
    parts = list(model_path.parts)
    if "outputs" in parts:
        start = parts.index("outputs") + 1
        relative_parts = [_sanitize_output_component(part) for part in parts[start:] if part not in ("", ".")]
        if relative_parts:
            return Path(*relative_parts)
    external_parts = [_sanitize_output_component(part) for part in model_name.split("/") if part.strip()]
    if not external_parts:
        external_parts = ["unknown_model"]
    return Path("external", *external_parts)


def resolve_task_output_paths(
    task: EvalTaskConfig,
    *,
    output_dir: Path,
    model_name: str,
) -> tuple[Path, Path]:
    if task.output_path is not None:
        output_root = Path(task.output_path)
        if output_root.suffix:
            raise ValueError("--output 必须是目录路径，不能是文件路径。")
    else:
        output_root = output_dir / build_model_output_path(model_name) / task.name
    result_output = output_root / "result.json"
    raw_output = task.raw_output_path or (output_root / "raw.json")
    return result_output, raw_output


def write_eval_result(path: str | Path, result: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return output_path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if callable(value):
        try:
            return getsource(value)
        except (OSError, TypeError):
            return str(value)
    return str(value)


def write_raw_eval_result(path: str | Path, result: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=False, indent=2)
    return output_path
