#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.sft_selection import scan_checkpoint_dirs, write_best_checkpoint_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用多个 vLLM 进程并行选择 SFT checkpoint。")
    parser.add_argument("--eval-config", required=True, help="评测配置文件路径。")
    parser.add_argument("--train-output-dir", required=True, help="SFT 训练输出目录。")
    parser.add_argument("--dataset", required=True, help="用于选择 checkpoint 的 dev 数据集路径。")
    parser.add_argument("--gpus", required=True, help="逗号分隔 GPU 列表，例如 0,1,2。")
    parser.add_argument("--backend", default=None, choices=("vllm",), help="覆盖评测后端。")
    parser.add_argument("--batch-size", default=None, help="覆盖 batch size。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖生成长度上限。")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条。")
    parser.add_argument("--output-subdir", default="dev_eval_parallel", help="合并 ranking 写入训练目录下的子目录。")
    parser.add_argument("--work-dir", default="tmp/select_sft_checkpoint_parallel", help="worker 临时目录。")
    return parser.parse_args()


def _gpu_list(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus 不能为空。")
    return gpus


def _shard_round_robin(items: list[Path], shard_count: int) -> list[list[Path]]:
    shards = [[] for _ in range(shard_count)]
    for index, item in enumerate(items):
        shards[index % shard_count].append(item)
    return [shard for shard in shards if shard]


def _safe_slug(path: str | Path) -> str:
    return Path(path).name.replace("/", "_")


def _prepare_worker_train_dir(worker_dir: Path, checkpoints: list[Path]) -> Path:
    train_dir = worker_dir / "train_output"
    train_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        link_path = train_dir / checkpoint.name
        if link_path.exists() or link_path.is_symlink():
            raise FileExistsError(f"worker checkpoint link 已存在：{link_path}")
        link_path.symlink_to(checkpoint.resolve(), target_is_directory=True)
    return train_dir


def _build_worker_command(args: argparse.Namespace, worker_train_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/select_sft_checkpoint.py"),
        "--eval-config",
        args.eval_config,
        "--train-output-dir",
        str(worker_train_dir),
        "--dataset",
        args.dataset,
    ]
    if args.backend is not None:
        command.extend(["--backend", args.backend])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def _copy_worker_eval_outputs(worker_ranking: dict, merged_dir: Path) -> None:
    for record in worker_ranking["records"]:
        checkpoint_name = Path(record["checkpoint_path"]).name
        source_dir = Path(record["result_path"]).parent
        target_dir = merged_dir / checkpoint_name / source_dir.name
        if target_dir.exists():
            raise FileExistsError(f"合并评测目录已存在：{target_dir}")
        shutil.copytree(source_dir, target_dir)
        record["result_path"] = str(target_dir / Path(record["result_path"]).name)
        record["raw_result_path"] = str(target_dir / Path(record["raw_result_path"]).name)


def main() -> None:
    args = parse_args()
    train_output_dir = Path(args.train_output_dir)
    checkpoints = scan_checkpoint_dirs(train_output_dir)
    if not checkpoints:
        raise ValueError(f"未在 {train_output_dir} 下找到 checkpoint-<step> 目录。")

    gpus = _gpu_list(args.gpus)
    shards = _shard_round_robin(checkpoints, min(len(gpus), len(checkpoints)))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    work_root = Path(args.work_dir) / f"{_safe_slug(train_output_dir)}_{run_id}"
    work_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[subprocess.Popen, Path, str, list[Path]]] = []
    for index, shard in enumerate(shards):
        gpu = gpus[index]
        worker_dir = work_root / f"worker{index}"
        worker_train_dir = _prepare_worker_train_dir(worker_dir, shard)
        command = _build_worker_command(args, worker_train_dir)
        log_path = worker_dir / "worker.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTHONPATH", str(ROOT / "src"))
        log_fh = log_path.open("w", encoding="utf-8")
        print(f"worker={index} gpu={gpu} checkpoints={[path.name for path in shard]} log={log_path}", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        processes.append((process, worker_dir, gpu, shard))

    failed: list[str] = []
    for process, worker_dir, gpu, _shard in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append(f"gpu={gpu} returncode={return_code} log={worker_dir / 'worker.log'}")
    if failed:
        raise RuntimeError("并行 checkpoint selection 有 worker 失败：\n" + "\n".join(failed))

    merged_dir = train_output_dir / args.output_subdir
    merged_dir.mkdir(parents=True, exist_ok=True)
    records = []
    checkpoint_by_name = {path.name: path for path in checkpoints}
    for _process, worker_dir, _gpu, _shard in processes:
        ranking_path = worker_dir / "train_output/dev_eval/dev_ranking.json"
        worker_ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        _copy_worker_eval_outputs(worker_ranking, merged_dir)
        for record in worker_ranking["records"]:
            checkpoint_name = Path(record["checkpoint_path"]).name
            record["checkpoint_path"] = str(checkpoint_by_name[checkpoint_name])
            records.append(record)

    records.sort(key=lambda record: int(record["global_step"]))
    ranking_path, best_path = write_best_checkpoint_summary(
        output_dir=merged_dir,
        dataset_path=args.dataset,
        records=records,
    )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    best["parallel_selection"] = {
        "gpus": gpus[: len(shards)],
        "worker_count": len(shards),
        "work_root": str(work_root),
    }
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"wrote_ranking={ranking_path}")
    print(f"wrote_best={best_path}")
    print(f"worker_root={work_root}")


if __name__ == "__main__":
    main()
