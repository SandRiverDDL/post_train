from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from post_train.io import ensure_parent

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "post_train"
RUN_INFO_NAME = "mlflow_run.json"


def _disabled() -> bool:
    if os.environ.get("LOCAL_RANK") not in {None, "", "0"}:
        return True
    return os.environ.get("POST_TRAIN_DISABLE_MLFLOW", "").lower() in {"1", "true", "yes", "on"}


def _mlflow():
    if _disabled():
        return None
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - 只在环境损坏时触发
        print(f"mlflow.disabled_reason={exc}")
        return None
    return mlflow


def _setup_mlflow():
    mlflow = _mlflow()
    if mlflow is None:
        return None
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME))
    return mlflow


def _repo_root() -> Path:
    return Path.cwd()


def _relative_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(_repo_root().resolve()))
    except Exception:
        return str(candidate)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_info() -> dict[str, Any]:
    dirty = bool(_git_value(["status", "--short"]))
    return {
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_dirty": dirty,
    }


def _flatten(value: Any, *, prefix: str = "", limit: int = 250) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(item, prefix=name, limit=limit))
        return out
    if isinstance(value, (list, tuple, set)):
        text = json.dumps(list(value), ensure_ascii=False, default=str)
        return {prefix: text[:limit]}
    if value is None or isinstance(value, (str, int, float, bool)):
        text = value if not isinstance(value, str) else value[:limit]
        return {prefix: text}
    return {prefix: str(value)[:limit]}


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        key_text = str(key).replace(" ", "_")[:250]
        if isinstance(value, bool):
            safe[key_text] = value
        elif isinstance(value, (int, float, str)):
            safe[key_text] = value
        else:
            safe[key_text] = str(value)[:250]
    return safe


def _safe_log_params(mlflow, params: dict[str, Any]) -> None:
    for key, value in _safe_params(params).items():
        try:
            mlflow.log_param(key, value)
        except Exception:
            # MLflow 的 param 不允许同一 run 内改值；评测写回训练 run 时直接保留旧值。
            continue


def _numeric_metrics(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(_numeric_metrics(value, prefix=name))
    return out


class MLflowRunContext(AbstractContextManager["MLflowRunContext"]):
    def __init__(
        self,
        *,
        run_name: str,
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.run_name = run_name
        self.tags = tags or {}
        self.params = params or {}
        self.output_dir = Path(output_dir) if output_dir else None
        self.config_path = Path(config_path) if config_path else None
        self.requested_run_id = run_id
        self.run_id: str | None = None
        self._mlflow = None
        self.active = False

    def __enter__(self) -> "MLflowRunContext":
        mlflow = _setup_mlflow()
        if mlflow is None:
            return self
        active = mlflow.active_run()
        if active is None:
            run = mlflow.start_run(run_id=self.requested_run_id, run_name=self.run_name)
            self.active = True
        else:
            run = active
        self._mlflow = mlflow
        self.run_id = run.info.run_id
        if self.tags:
            tags = {str(k): str(v) for k, v in self.tags.items() if v is not None}
            if self.requested_run_id:
                tags.pop("route", None)
            mlflow.set_tags(tags)
        params = _safe_params({**git_info(), **self.params})
        if params:
            _safe_log_params(mlflow, params)
        if self.output_dir:
            write_run_info(self.output_dir, self.run_id)
        if self.config_path and self.config_path.exists():
            mlflow.log_artifact(str(self.config_path), artifact_path="config")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._mlflow is None:
            return
        if exc_type is not None:
            self._mlflow.set_tag("status", "failed")
            self._mlflow.set_tag("error", str(exc)[:500])
        else:
            self._mlflow.set_tag("status", "finished")
        if self.active:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is not None:
            _safe_log_params(self._mlflow, params)

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self._mlflow is not None:
            numeric = _numeric_metrics(metrics)
            if numeric:
                self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)


def start_run(
    *,
    run_name: str,
    route: str,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    params: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> MLflowRunContext:
    base_params = {
        "route": route,
        "config_path": _relative_path(config_path),
        "output_dir": _relative_path(output_dir),
    }
    if params:
        base_params.update(_flatten(params))
    return MLflowRunContext(
        run_name=run_name,
        tags={"route": route, "status": "running"},
        params=base_params,
        output_dir=output_dir,
        config_path=config_path,
        run_id=run_id,
    )


def write_run_info(output_dir: str | Path, run_id: str) -> Path:
    path = ensure_parent(Path(output_dir) / RUN_INFO_NAME)
    payload = {
        "run_id": run_id,
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        "experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def find_run_id_for_model(model_path: str | Path) -> str | None:
    current = Path(model_path)
    candidates = [current]
    if current.name.startswith("checkpoint-"):
        candidates.append(current.parent)
    for candidate in candidates:
        info_path = candidate / RUN_INFO_NAME
        if info_path.exists():
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            return str(payload.get("run_id") or "") or None
    return None


def _log_json_artifact(payload: dict[str, Any], *, filename: str, artifact_path: str) -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_training_summary(summary: dict[str, Any]) -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.set_tag("status", str(summary.get("status", "finished")))
    _safe_log_params(
        mlflow,
        {
            "train_dataset": summary.get("train_dataset"),
            "base_model": summary.get("base_model"),
            "parent_run_id": summary.get("parent_run_id"),
            "run_summary_path": summary.get("summary_path"),
        },
    )
    if summary.get("dev_metrics"):
        mlflow.log_metrics(_numeric_metrics(summary["dev_metrics"], prefix="dev"))
    _log_json_artifact(summary, filename="run_summary.json", artifact_path="summary")


def log_eval_result(run_result: dict[str, Any]) -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    result = dict(run_result.get("result") or {})
    task_name = str(run_result.get("task_name") or result.get("task_name") or "eval")
    metrics = dict(result.get("metrics") or {})
    mlflow.log_metrics(_numeric_metrics(metrics, prefix=f"eval/{task_name}"))
    _safe_log_params(
        mlflow,
        {
            f"eval/{task_name}/dataset": result.get("dataset_path"),
            f"eval/{task_name}/result_path": _relative_path(run_result.get("result_path")),
            f"eval/{task_name}/raw_result_path": _relative_path(run_result.get("raw_result_path")),
        },
    )
    result_path = Path(str(run_result.get("result_path", "")))
    if result_path.exists():
        mlflow.log_artifact(str(result_path), artifact_path=f"eval/{task_name}")


def log_data_result(result: dict[str, Any], *, artifact_name: str = "data_result.json") -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    outputs = dict(result.get("outputs") or {})
    report = dict(result.get("report") or result.get("summary") or {})
    _safe_log_params(mlflow, {f"data_path/{k}": _relative_path(v) for k, v in outputs.items()})
    mlflow.log_metrics(_numeric_metrics(report, prefix="data"))
    _log_json_artifact(result, filename=artifact_name, artifact_path="data")


def log_data_artifacts(paths: list[str | Path], *, artifact_path: str = "data") -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size < 20 * 1024 * 1024:
            mlflow.log_artifact(str(candidate), artifact_path=artifact_path)


def copy_config_artifact(config_path: str | Path | None) -> None:
    mlflow = _setup_mlflow()
    if mlflow is None or mlflow.active_run() is None or config_path is None:
        return
    path = Path(config_path)
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path="config")


def create_trainer_callback():
    mlflow = _setup_mlflow()
    if mlflow is None:
        return None
    try:
        from transformers import TrainerCallback
    except Exception:
        return None

    class MLflowTrainerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if mlflow.active_run() is None:
                return control
            if logs and state.global_step > 0:
                numeric = {key: float(value) for key, value in logs.items() if isinstance(value, (int, float))}
                if numeric:
                    mlflow.log_metrics(numeric, step=int(state.global_step))
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if mlflow.active_run() is not None:
                mlflow.set_tag("status", "finished")
            return control

    return MLflowTrainerCallback()
