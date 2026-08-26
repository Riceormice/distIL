#!/usr/bin/env python3
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def require(path: str, *snippets: str) -> str:
    source = (REPO / path).read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in source:
            raise AssertionError(f"{path}: missing protocol invariant: {snippet!r}")
    return source


def main() -> None:
    grpo_4b = require(
        "scripts/math/run_grpo_4b.sh",
        "run_verl_method_a800_4b.sh\" grpo",
    )
    grpo_8b = require(
        "scripts/math/run_grpo_8b.sh",
        "MODEL_SIZE=8b",
        "HARDWARE=h200",
        "run_verl_method_h200.sh\" grpo",
    )
    for path, source in (("run_grpo_4b.sh", grpo_4b), ("run_grpo_8b.sh", grpo_8b)):
        if "grpo_train.py" in source or "accelerate launch" in source:
            raise AssertionError(f"{path}: legacy TRL GRPO entrypoint is still reachable")

    require(
        "scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh",
        "VAL_N=16",
        "MAX_STEPS=100",
        "SCHEDULER_HORIZON_STEPS=420",
        "TRAIN_BATCH_SIZE=8",
        "PPO_MINI_BATCH_SIZE=8",
        "ROLLOUT_N=8",
        "8b:grpo) EVAL_MAX_NEW_TOKENS=16384",
        "learning_rate=5e-6",
        "lr_scheduler=linear",
        "warmup_steps=0",
        "max_response_length=16384",
        "training_temperature=0.7",
        "training_top_p=0.95",
        "training_top_k=20",
        "evaluation_frequency=5",
        "evaluation_samples_per_question=${VAL_N}",
        "evaluation_submission_mode=${EVAL_SUBMISSION_MODE}",
        "lock_protocol_file",
        "Merged checkpoint/tokenizer preflight: PASS",
    )

    require(
        "scripts/math/opsd_grouped8x8_a800_4b/a800_opsd_grouped8x8.sh",
        "MODEL_SIZE=4b",
        "HARDWARE=a800",
        "MAX_STEPS=\"${MAX_STEPS:-100}\"",
        "SCHEDULER_HORIZON_STEPS=\"${SCHEDULER_HORIZON_STEPS:-420}\"",
        "EVAL_FREQUENCY=\"${EVAL_FREQUENCY:-5}\"",
        "VAL_N=\"${VAL_N:-16}\"",
        "GRADIENT_ACCUMULATION_STEPS=8",
        "GROUPED_UNIQUE_PROMPTS_PER_STEP=8",
        "GROUPED_ROLLOUTS_PER_PROMPT=8",
        "LR=5e-6",
        "MAX_COMPLETION_LENGTH=16384",
        "JSD_TOKEN_CLIP=0.05",
        "EVAL_SUBMISSION_MODE=\"${EVAL_SUBMISSION_MODE:-legacy_all_prompts}\"",
    )

    require(
        "scripts/math/train_eval5_n16_h200/run_distil_method_h200.sh",
        "8b:h200",
        "JSD_TOKEN_CLIP=\"${JSD_TOKEN_CLIP:-0.06}\"",
        "VLLM_GPU_MEMORY_UTILIZATION=\"${VLLM_GPU_MEMORY_UTILIZATION:-0.42}\"",
        "EVAL_TEMPERATURE=1.0",
        "EVAL_TOP_P=1.0",
        "EVAL_TOP_K=-1",
        "EVAL_MAX_NEW_TOKENS=38912",
        "4b:a800",
        "JSD_TOKEN_CLIP=\"${JSD_TOKEN_CLIP:-0.05}\"",
        "VLLM_GPU_MEMORY_UTILIZATION=\"${VLLM_GPU_MEMORY_UTILIZATION:-0.45}\"",
        "unique_prompts_per_optimizer_step=${UNIQUE_PROMPTS_PER_OPTIMIZER_STEP}",
        "rollouts_per_question=${ROLLOUTS_PER_QUESTION}",
        "training_trajectories_per_optimizer_step=${TRAJECTORIES_PER_OPTIMIZER_STEP}",
        "EVAL_TEMPERATURE=0.7",
        "EVAL_TOP_P=0.95",
        "EVAL_TOP_K=20",
        "EVAL_MAX_NEW_TOKENS=16384",
        "MODEL_SIZE=\"${MODEL_SIZE}\"",
        "lock_protocol_file",
    )

    print("GRPO/OPSD launcher protocol invariants: PASS")


if __name__ == "__main__":
    main()
