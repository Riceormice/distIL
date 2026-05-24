#!/bin/bash
# scripts/coding/run_distil.sh
# Train DistIL on LiveCodeBench (LCBv6) with rich execution feedback.
# Paper: Table 4 (Coding column) hyperparameters.
#
# Usage:
#   bash scripts/coding/run_distil.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run] Commands will be printed but not executed."

# =============================================================================
# CLUSTER CONFIGURATION — edit these for your cluster
# =============================================================================
ACCOUNT="${ACCOUNT:-your_account}"
PARTITION="${PARTITION:-gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"         # 2 × H200 in paper
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-256000}"
REPO_DIR="${REPO_DIR:-/path/to/distil/SDPO}"

# =============================================================================
# EXPERIMENT CONFIGURATION (Table 4, Coding column)
# =============================================================================
CONFIG_NAME="distil"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATA_PATH="datasets/lcb_v6"

TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=1
ROLLOUT_N=8
LR="1e-6"
LR_WARMUP_STEPS=0
DISTILLATION_TOPK=20
TEACHER_UPDATE_RATE=0.01
VAL_N=4
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95

# =============================================================================
# JOB SUBMISSION
# =============================================================================
MODEL_NAME=$(echo "${MODEL_PATH}" | tr '/' '-')
EXP_NAME="DistIL-coding-lcbv6-${MODEL_NAME}"

args="data.train_batch_size=${TRAIN_BATCH_SIZE} \
trainer.group_name=DistIL-coding \
actor_rollout_ref.rollout.n=${ROLLOUT_N} \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.use_reference_solution=False \
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
    --job-name="distil-coding"
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
    echo "[dry-run] exp: ${EXP_NAME}"
    echo "${sbatch_args[@]}"
else
    echo "Submitting: ${EXP_NAME}"
    "${sbatch_args[@]}"
fi
