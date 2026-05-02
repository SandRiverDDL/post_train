"""
占卡脚本：占用指定 GPU 的显存与算力，防止被别人抢占。
用法：
    python hold_gpu.py                  # 占用所有可见 GPU，显存占 90%
    python hold_gpu.py --gpus 0,1       # 只占 0、1 号卡
    python hold_gpu.py --mem 0.8        # 占 80% 显存
    python hold_gpu.py --compute light  # 轻量算力占用（默认）
    python hold_gpu.py --compute heavy  # 重度算力占用（跑满 SM）

按 Ctrl+C 即可释放。
"""

import argparse
import os
import time
import signal
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default=None,
                   help="要占用的 GPU id，逗号分隔。默认占全部可见 GPU。")
    p.add_argument("--mem", type=float, default=0.9,
                   help="每张卡占用的显存比例 (0~1)，默认 0.9。")
    p.add_argument("--compute", choices=["none", "light", "heavy"],
                   default="light",
                   help="算力占用模式：none 只占显存；light 低频小矩阵乘（默认）；heavy 持续跑大矩阵乘。")
    p.add_argument("--interval", type=float, default=2.0,
                   help="light 模式下每次计算的间隔秒数，默认 2s。")
    return p.parse_args()


def get_gpu_ids(arg_gpus):
    if arg_gpus is not None:
        return [int(x) for x in arg_gpus.split(",") if x.strip() != ""]
    return list(range(torch.cuda.device_count()))


def allocate_memory(device, ratio):
    """在给定 device 上占用 ratio 比例的显存，返回持有的 tensor。"""
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    target = int(total * ratio)

    target = min(target, int(free * 0.98))

    elem_size = 4
    numel = target // elem_size

    chunks = []
    remaining = numel
    chunk_size = 1 << 28
    while remaining > 0:
        n = min(chunk_size, remaining)
        try:
            t = torch.empty(n, dtype=torch.float32, device=device)
            chunks.append(t)
            remaining -= n
        except RuntimeError:
            chunk_size //= 2
            if chunk_size < (1 << 20):
                break
    allocated = sum(c.numel() * elem_size for c in chunks)
    print(f"[GPU {device}] 已占用显存: {allocated/1024**3:.2f} GiB / 总 {total/1024**3:.2f} GiB")
    return chunks


def setup_signal_handler():
    def handler(sig, frame):
        print("\n收到退出信号，释放显存并退出。")
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("未检测到可用的 CUDA 设备。")
        return

    gpu_ids = get_gpu_ids(args.gpus)
    print(f"将占用 GPU: {gpu_ids}，显存比例: {args.mem}，算力模式: {args.compute}")
    print(f"当前 PID: {os.getpid()}，按 Ctrl+C 释放。")

    setup_signal_handler()

    held_tensors = {}
    compute_tensors = {}
    for gid in gpu_ids:
        held_tensors[gid] = allocate_memory(gid, args.mem)
        if args.compute != "none":
            size = 4096 if args.compute == "heavy" else 1024
            a = torch.randn(size, size, device=gid)
            b = torch.randn(size, size, device=gid)
            compute_tensors[gid] = (a, b)

    print("占卡中... (Ctrl+C 释放)")
    try:
        while True:
            if args.compute == "heavy":
                for gid, (a, b) in compute_tensors.items():
                    with torch.cuda.device(gid):
                        c = a @ b
                        a.copy_(c)
            elif args.compute == "light":
                for gid, (a, b) in compute_tensors.items():
                    with torch.cuda.device(gid):
                        _ = a @ b
                        torch.cuda.synchronize(gid)
                time.sleep(args.interval)
            else:
                time.sleep(60)
    except KeyboardInterrupt:
        print("\n已释放。")


if __name__ == "__main__":
    main()
