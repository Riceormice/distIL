#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from grouped_repeat_sampler import GroupedRepeatSampler


EXPECTED_DATASET_SHA256 = "cd691ec524933a3828d117b744cf57b9662bd189b4a1cd9192d897ec4d0614ed"


def load_and_verify_dataset(path: Path, expected_size: int, expected_sha256: str) -> int:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"dataset SHA256 mismatch: expected {expected_sha256}, found {digest}: {path}"
        )

    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected_size:
        raise SystemExit(f"expected {expected_size} dataset rows, found {len(rows)}: {path}")

    problems = [row.get("problem") for row in rows]
    if any(not isinstance(problem, str) or not problem.strip() for problem in problems):
        raise SystemExit(f"dataset contains a missing or empty problem field: {path}")
    if len(set(problems)) != expected_size:
        raise SystemExit(
            f"expected {expected_size} unique problems, found {len(set(problems))}: {path}"
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-size", type=int, default=758)
    parser.add_argument("--expected-dataset-sha256", default=EXPECTED_DATASET_SHA256)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.per_device_batch_size != 1:
        raise SystemExit("this verifier currently requires per-device batch size 1")
    unique_prompts = args.world_size * args.per_device_batch_size
    if args.gradient_accumulation_steps != args.rollouts_per_prompt:
        raise SystemExit("gradient accumulation must equal rollouts per prompt")

    dataset_size = load_and_verify_dataset(
        args.dataset,
        expected_size=args.expected_dataset_size,
        expected_sha256=args.expected_dataset_sha256,
    )

    sampler = GroupedRepeatSampler(
        dataset_size,
        unique_prompts_per_step=unique_prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        seed=args.seed,
    )
    global_indices = list(sampler)
    local_indices = [global_indices[rank::args.world_size] for rank in range(args.world_size)]

    for optimizer_step in range(sampler.groups_per_epoch):
        start = optimizer_step * args.gradient_accumulation_steps * args.per_device_batch_size
        stop = start + args.gradient_accumulation_steps * args.per_device_batch_size
        prompts = []
        for rank in range(args.world_size):
            local_block = local_indices[rank][start:stop]
            if len(set(local_block)) != args.per_device_batch_size:
                raise RuntimeError(
                    f"rank {rank}, optimizer step {optimizer_step}: prompt changed inside rollout group: {local_block}"
                )
            prompts.extend(local_block[: args.per_device_batch_size])
        if len(set(prompts)) != unique_prompts:
            raise RuntimeError(f"optimizer step {optimizer_step}: prompts are not unique: {prompts}")

    first_group = next(sampler.grouped_indices())
    print("GROUPED 8x8 SAMPLER: PASS")
    print(f"dataset={args.dataset.resolve()}")
    print(f"dataset_rows={dataset_size}")
    print(f"unique_problems={dataset_size}")
    print(f"dataset_sha256={args.expected_dataset_sha256}")
    print(f"unique_prompts_per_optimizer_step={unique_prompts}")
    print(f"rollouts_per_prompt={args.rollouts_per_prompt}")
    print(f"trajectories_per_optimizer_step={unique_prompts * args.rollouts_per_prompt}")
    print(f"groups_per_epoch={sampler.groups_per_epoch}")
    print(f"first_prompt_indices={list(first_group)}")


if __name__ == "__main__":
    main()
