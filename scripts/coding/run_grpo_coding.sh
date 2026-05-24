#!/bin/bash
# scripts/coding/run_grpo.sh
# Train GRPO baseline on LiveCodeBench (LCBv6).
# Paper: Table 5 hyperparameters.
#
# Usage:
#   bash scripts/coding/run_grpo.sh [--dry-run]
#
# Set GRPO_MODE=off_policy (default) or GRPO_MODE=on_policy to switch variants.

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run] Commands will be printed but not executed."

# =============================================================================
# CLUSTER CONFIGURATION — edit these for your cluster
# =============================================================================
ACCOUNT="${ACCOUNT:-your_account}"
PARTITION="${PARTITION:-gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-256000}"
REPO_DIR="${REPO_DIR:-/path/to/distil/SDPO}"

# =============================================================================
# EXPERIMENT CONFIGURATION (Table 5 of paper)
# =============================================================================
CONFIG_NAME="ppo_trainer"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATA_PATH="datasets/lcb_v6"
GRPO_MODE="${GRPO_MODE:-off_policy}"   # off_policy | on_policy

TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
VAL_N=4
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95

if [[ "${GRPO_MODE}" == "on_policy" ]]; then
    MINI_BATCH_SIZE=32
    LR="1e-5"
    LR_WARMUP_STEPS=10
else
    MINI_BATCH_SIZE=8
    LR="1e-6"
    LR_WARMUP_STEPS=10
fi

# =============================================================================
# JOB SUBMISSION
# =============================================================================
MODEL_NAME=$(echo "${MODEL_PATH}" | tr '/' '-')
EXP_NAME="GRPO-${GRPO_MODE}-coding-lcbv6-${MODEL_NAME}"

args="data.train_batch_size=${TRAIN_BATCH_SIZE} \
trainer.group_name=GRPO-coding-${GRPO_MODE} \
actor_rollout_ref.rollout.n=${ROLLOUT_N} \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
actor_rollout_ref.actor.clip_ratio_high=0.28 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
trainer.n_gpus_per_node=${GPUS_PER_NODE}"

run_cmd="bash ${REPO_DIR}/training/verl_training.sh ${EXP_NAME} ${CONFIG_NAME} ${DATA_PATH} ${args}"
setup_cmd="pip install -e ${REPO_DIR} --quiet; export PYTHONPATH=${REPO_DIR}:\$PYTHONPATH"
wrapped="srun bash -c '${setup_cmd}; ${run_cmd}'"

mkdir -p "${REPO_DIR}/../logs"

sbatch_args=(
    sbatch
    --job-name="grpo-coding"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=1
    --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${REPO_DIR}/../logs/${EXP_NAME}_%j.log"
    --error="${REPO_DIR}/../logs/${EXP_NAME}_%j.err"
    --wrap="${wrapped}"
)

if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] exp: ${EXP_NAME} | mode: ${GRPO_MODE}"
    echo "${sbatch_args[@]}"
else
    echo "Submitting: ${EXP_NAME}"
    "${sbatch_args[@]}"
fi
