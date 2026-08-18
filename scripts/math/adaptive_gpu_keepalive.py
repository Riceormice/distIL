#!/usr/bin/env python3
"""Add short GPU bursts only while the experiment workload is CPU-bound."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--minimum-utilization", type=int, default=38)
    parser.add_argument("--burst-seconds", type=float, default=0.7)
    parser.add_argument("--idle-seconds", type=float, default=0.8)
    parser.add_argument("--startup-delay-seconds", type=float, default=30.0)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--minimum-used-memory-mib", type=int, default=4096)
    return parser.parse_args()


def parent_is_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def should_stop(args: argparse.Namespace) -> bool:
    return args.stop_file.exists() or not parent_is_alive(args.parent_pid)


def read_gpu_stats(physical_gpu: str) -> tuple[int, int] | None:
    command = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        utilization, used_memory = output.strip().splitlines()[0].split(",")
        return int(utilization.strip()), int(used_memory.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_or_stopped(args: argparse.Namespace, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if should_stop(args):
            return True
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return should_stop(args)


def main() -> None:
    args = parse_args()
    if wait_or_stopped(args, args.startup_delay_seconds):
        return

    # Wait until the real workload has initialized CUDA. Starting earlier would
    # change the amount of free memory measured by vLLM during engine startup.
    while not should_stop(args):
        stats = read_gpu_stats(args.physical_gpu)
        if stats is not None and stats[1] >= args.minimum_used_memory_mib:
            break
        if wait_or_stopped(args, 5.0):
            return

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "A keepalive worker must see exactly one CUDA device; "
            f"found {torch.cuda.device_count()}."
        )

    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    size = args.matrix_size
    try:
        left = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((size, size), device="cuda", dtype=torch.bfloat16)
        torch.mm(left, right, out=output)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        print(
            f"keepalive disabled on physical_gpu={args.physical_gpu}: {exc}",
            flush=True,
        )
        return

    print(
        "adaptive GPU keepalive started: "
        f"physical_gpu={args.physical_gpu}, "
        f"minimum_utilization={args.minimum_utilization}%, matrix_size={size}",
        flush=True,
    )

    with torch.inference_mode():
        while not should_stop(args):
            stats = read_gpu_stats(args.physical_gpu)
            if stats is not None and stats[0] < args.minimum_utilization:
                deadline = time.monotonic() + args.burst_seconds
                while time.monotonic() < deadline and not should_stop(args):
                    torch.mm(left, right, out=output)
                    torch.cuda.synchronize()
            if wait_or_stopped(args, args.idle_seconds):
                break

    print(
        f"adaptive GPU keepalive stopped: physical_gpu={args.physical_gpu}",
        flush=True,
    )


if __name__ == "__main__":
    main()
