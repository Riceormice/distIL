#!/usr/bin/env python3
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
DIRECTORY = REPO / "scripts/math/sweeps/sr_opsd_alpha_rho_4b_a800"


def require(path: Path, *snippets: str) -> str:
    source = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in source:
            raise AssertionError(f"{path}: missing invariant {snippet!r}")
    return source


def main() -> None:
    runner = require(
        DIRECTORY / "run_sr_opsd_4b_alpha_rho_a800.sh",
        "0.9,0.7|0.9,0.9|0.7,0.7|0.7,0.9|0.7,0.95",
        "MODEL_SIZE=4b",
        "HARDWARE=a800",
        "DIVERGENCE_ALPHA=0.25",
        "EVAL_SUBMISSION_MODE=legacy_all_prompts",
        "EVAL_PROMPT_BATCH_SIZE=0",
        "sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829",
        "run_verl_method_a800_4b.sh\" sr_opsd",
    )
    if "Qwen3-4B-Instruct" in runner:
        raise AssertionError("the sweep must use the established dense Qwen3-4B model")

    points = {
        "a800_alpha070_rho070.sh": "0.7 0.7",
        "a800_alpha070_rho090.sh": "0.7 0.9",
        "a800_alpha070_rho095.sh": "0.7 0.95",
        "a800_alpha090_rho070.sh": "0.9 0.7",
        "a800_alpha090_rho090.sh": "0.9 0.9",
    }
    for filename, point in points.items():
        require(DIRECTORY / filename, "run_sr_opsd_4b_alpha_rho_a800.sh", point)

    native = require(
        REPO / "scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh",
        "VAL_N=16",
        "MAX_STEPS=100",
        "SCHEDULER_HORIZON_STEPS=420",
        "TRAIN_BATCH_SIZE=8",
        "PPO_MINI_BATCH_SIZE=8",
        "ROLLOUT_N=8",
        "4b:*) EVAL_MAX_NEW_TOKENS=16384",
        "4b)",
        "EVAL_TEMPERATURE=0.7",
        "EVAL_TOP_P=0.95",
        "EVAL_TOP_K=20",
        "SAVE_FREQ=5",
        'RUN_NAME="sr-opsd-${MODEL_SIZE}-seed0-native-verl-forward-renyi-rho${RENYI_ORDER}-refw${SELF_REFERENCE_WEIGHT}',
    )
    if "Qwen3-4B-Instruct-2507" in native:
        raise AssertionError("the native sweep runner unexpectedly names an instruct model")

    require(
        REPO / "SDPO/run_local_math_verl.sh",
        '"trainer.save_freq=${SAVE_FREQ}"',
        '"trainer.resume_mode=auto"',
        '"actor_rollout_ref.actor.self_distillation.rho=${RENYI_ORDER}"',
        '"actor_rollout_ref.actor.self_distillation.renyi_regularization_level=${SELF_REFERENCE_WEIGHT}"',
    )

    nightly = require(
        REPO / "scripts/nightly/run_current_experiment.sh",
        "math_4b_alpha070_rho070",
        "math_4b_alpha070_rho090",
        "math_4b_alpha070_rho095",
        "math_4b_alpha090_rho070",
        "math_4b_alpha090_rho090",
    )
    if nightly.count("math_4b_alpha") != 10:
        raise AssertionError("each 4B point must appear once in the job list and once in dispatch")

    require(
        REPO / "scripts/nightly/show_current_progress.py",
        "Math 4B a=.7 r=.7",
        "Math 4B a=.9 r=.9",
        "sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829",
    )
    print("Qwen3-4B Math alpha/rho sweep launch invariants: PASS")


if __name__ == "__main__":
    main()
