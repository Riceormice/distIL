#!/usr/bin/env python3
"""Verify that a math runtime is complete and internally self-contained."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("verl", "opsd"), required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gpu-smoke", action="store_true")
    return parser.parse_args()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def import_from_prefix(name: str, prefix: Path):
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{name} has no module file")
    path = Path(module_file).resolve()
    if not is_within(path, prefix):
        raise RuntimeError(f"{name} leaked from outside the environment: {path}")
    print(f"{name}={path}")
    return module


def main() -> None:
    args = parse_args()
    prefix = args.prefix.resolve()
    repo = args.repo.resolve()

    if not is_within(Path(sys.executable), prefix):
        raise RuntimeError(f"Python executable is outside the environment: {sys.executable}")
    if Path(sys.prefix).resolve() != prefix:
        raise RuntimeError(f"sys.prefix mismatch: expected {prefix}, found {sys.prefix}")
    if os.environ.get("PYTHONHOME"):
        raise RuntimeError("PYTHONHOME must be unset")

    torch = import_from_prefix("torch", prefix)
    import_from_prefix("transformers", prefix)
    import_from_prefix("tokenizers", prefix)
    import_from_prefix("datasets", prefix)
    ray = import_from_prefix("ray", prefix)
    vllm = import_from_prefix("vllm", prefix)
    import_from_prefix("vllm._C", prefix)
    flash_attn = import_from_prefix("flash_attn", prefix)
    import_from_prefix("flash_attn_2_cuda", prefix)
    import_from_prefix("accelerate", prefix)
    import_from_prefix("peft", prefix)
    import_from_prefix("math_verify", prefix)

    manager = Path(torch.__file__).resolve().parent / "bin/torch_shm_manager"
    if not manager.is_file() or not os.access(manager, os.X_OK):
        raise RuntimeError(f"torch_shm_manager is missing or not executable: {manager}")

    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")
    torch.zeros(1).share_memory_()

    if args.profile == "verl":
        expected = {"torch": "2.7.1", "vllm": "0.10.0", "transformers": "4.57.1"}
        sys.path.insert(0, str(repo / "SDPO"))
        verl = importlib.import_module("verl")
        if not is_within(Path(verl.__file__), repo / "SDPO"):
            raise RuntimeError(f"verl did not load from the repository: {verl.__file__}")
        from vllm.v1.engine.utils import CoreEngineProcManager  # noqa: F401
    else:
        expected = {"torch": "2.8.0", "vllm": "0.11.0", "transformers": "4.57.1"}
        import_from_prefix("trl", prefix)
        import_from_prefix("deepspeed", prefix)
        import_from_prefix("bitsandbytes", prefix)
        import_from_prefix("xformers", prefix)

    actual = {
        "torch": torch.__version__.split("+")[0],
        "vllm": vllm.__version__,
        "transformers": importlib.import_module("transformers").__version__,
    }
    if actual != expected:
        raise RuntimeError(f"version mismatch: expected {expected}, found {actual}")
    expected_cuda = "12.6" if args.profile == "verl" else "12.8"
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"CUDA runtime mismatch: expected {expected_cuda}, found {torch.version.cuda}")
    if flash_attn.__version__ != "2.8.3":
        raise RuntimeError(f"FlashAttention mismatch: expected 2.8.3, found {flash_attn.__version__}")

    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    try:
        @ray.remote
        def worker_origins() -> dict[str, str]:
            import torch as worker_torch
            import transformers as worker_transformers
            import vllm as worker_vllm

            return {
                "torch": worker_torch.__file__,
                "transformers": worker_transformers.__file__,
                "vllm": worker_vllm.__file__,
            }

        for name, module_file in ray.get(worker_origins.remote()).items():
            if not is_within(Path(module_file), prefix):
                raise RuntimeError(f"Ray worker loaded {name} outside the environment: {module_file}")
            print(f"ray_worker.{name}={Path(module_file).resolve()}")
    finally:
        ray.shutdown()

    if args.gpu_smoke:
        if torch.cuda.device_count() < 1:
            raise RuntimeError("GPU smoke test requested but CUDA has no visible devices")
        capability = torch.cuda.get_device_capability(0)
        if capability not in {(8, 0), (9, 0)}:
            raise RuntimeError(f"unsupported GPU capability: {capability}")
        from flash_attn import flash_attn_func

        q = torch.randn((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
        out = flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
        if out.shape != q.shape:
            raise RuntimeError(f"FlashAttention output shape mismatch: {out.shape} != {q.shape}")
        print(f"gpu={torch.cuda.get_device_name(0)} capability={capability}")

    print(f"profile={args.profile}")
    print(f"prefix={prefix}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"vllm={vllm.__version__}")
    print(f"flash_attn={flash_attn.__version__}")
    print("UNIFIED MATH ENVIRONMENT: PASS")


if __name__ == "__main__":
    main()
