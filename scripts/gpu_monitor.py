#!/usr/bin/env python3
"""实时监控 GPU 利用率与显存，并给出简要优化建议。"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class GpuSample:
    util_gpu: float
    util_mem: float
    mem_used_mib: float
    mem_total_mib: float
    temperature: float
    power_w: float


def query_gpu() -> list[GpuSample]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"无法调用 nvidia-smi: {exc}") from exc

    samples: list[GpuSample] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        samples.append(
            GpuSample(
                util_gpu=float(parts[1]),
                util_mem=float(parts[2]),
                mem_used_mib=float(parts[3]),
                mem_total_mib=float(parts[4]),
                temperature=float(parts[5]),
                power_w=float(parts[6]),
            )
        )
    return samples


def query_processes() -> list[tuple[str, str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    rows: list[tuple[str, str, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append((parts[1], parts[2], float(parts[3])))
    return rows


def format_bar(ratio: float, width: int = 24) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def advise(util_avg: float, mem_ratio_avg: float) -> str:
    if mem_ratio_avg < 0.35 and util_avg < 70:
        return "显存与算力均偏低：可增大 dynamic_batch_size_schedule 或提高 prefetch_factor"
    if mem_ratio_avg < 0.35:
        return "显存偏低：可增大 batch_size / dynamic_batch_size_schedule"
    if util_avg < 65:
        return "GPU 利用率偏低：检查 DataLoader（num_workers/prefetch）、或减少 empty_cache 频率"
    if mem_ratio_avg > 0.92:
        return "显存接近上限：减小 batch_size 或开启 gradient_accumulation_steps"
    return "资源利用较均衡"


def main() -> None:
    parser = argparse.ArgumentParser(description="监控 GPU 利用率与显存")
    parser.add_argument("-i", "--interval", type=float, default=2.0, help="采样间隔（秒）")
    parser.add_argument("-n", "--count", type=int, default=0, help="采样次数，0 表示持续运行")
    parser.add_argument("--window", type=int, default=15, help="滑动平均窗口大小")
    parser.add_argument("--gpu", type=int, default=0, help="显示的 GPU 索引")
    args = parser.parse_args()

    util_hist: deque[float] = deque(maxlen=max(1, args.window))
    mem_hist: deque[float] = deque(maxlen=max(1, args.window))

    iteration = 0
    try:
        while True:
            gpus = query_gpu()
            if not gpus:
                print("未检测到 GPU", file=sys.stderr)
                break
            if args.gpu >= len(gpus):
                raise SystemExit(f"GPU 索引 {args.gpu} 不存在（共 {len(gpus)} 张卡）")

            g = gpus[args.gpu]
            mem_ratio = g.mem_used_mib / max(g.mem_total_mib, 1.0)
            util_hist.append(g.util_gpu)
            mem_hist.append(mem_ratio)
            util_avg = sum(util_hist) / len(util_hist)
            mem_avg = sum(mem_hist) / len(mem_hist)

            ts = time.strftime("%H:%M:%S")
            print(
                f"[{ts}] GPU{args.gpu} "
                f"util={g.util_gpu:5.1f}% (avg {util_avg:5.1f}%) "
                f"mem={g.mem_used_mib/1024:6.1f}/{g.mem_total_mib/1024:6.1f} GiB "
                f"({mem_ratio*100:5.1f}%, avg {mem_avg*100:5.1f}%) "
                f"{format_bar(mem_ratio)} "
                f"T={g.temperature:.0f}C P={g.power_w:.0f}W"
            )
            procs = query_processes()
            if procs:
                for pid, name, mem in procs:
                    print(f"         pid={pid} {name} {mem:.0f} MiB")

            if iteration == 0 or (args.count > 0 and iteration == args.count - 1):
                print(f"         hint: {advise(util_avg, mem_avg)}")

            iteration += 1
            if args.count > 0 and iteration >= args.count:
                break
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        print("\n[stopped]")


if __name__ == "__main__":
    main()
