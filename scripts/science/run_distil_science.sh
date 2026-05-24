#!/bin/bash
# scripts/science/run_distil.sh
# Train DistIL on scientific reasoning benchmarks (SciKnowEval L3).
# Paper: Table 4 hyperparameters.
#
# Usage:
#   bash scripts/science/run_distil.sh [--dry-run]
#
# Override defaults via environment variables, e.g.:
#   MODELS="Qwen/Qwen3-8B" PARTITION=gpu bash scripts/science/run_distil.sh

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run] Commands will be printed but not executed."

# =============================================================================
# CLUSTER CONFIGURATION — edit these for your cluster
# =============================================================================
ACCOUNT="${ACCOUNT:-your_account}"           # SLURM account
PARTITION="${PARTITION:-gpu}"                # SLURM partition
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"         # GPUs per node (4 × H200 in paper)
TIME="${TIME:-12:00:00}"                     # Wall-clock limit
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-460000}"                         # MB
REPO_DIR="${REPO_DIR:-/path/to/distil/SDPO}" # Absolute path to SDPO subdir

# =============================================================================
# EXPERIMENT CONFIGURATION (Table 4 of paper)
# =============================================================================
CONFIG_NAME="distil"

MODELS=(
    ${MODELS:-"Qwen/Qwen3-8B allenai/Olmo-3-7B-Instruct"}
)

DATA_PATHS=(
    "datasets/sciknoweval/biology"
    "datasets/sciknoweval/chemistry"
    "datasets/sciknoweval/material"
    "datasets/sciknoweval/physics"
)

TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
ROLLOUT_N=8
LR="5e-5"
LR_WARMUP_STEPS=10
DISTILLATION_TOPK=100
TEACHER_UPDATE_RATE=0.01
VAL_N=16
VAL_TEMPERATURE=0.6

# =============================================================================
# JOB SUBMISSION
# =============================================================================
submit_job() {
    local exp_name="$1"
    local data_path="$2"
    local model_path="$3"

    local args="data.train_batch_size=${TRAIN_BATCH_SIZE} \
trainer.group_name=DistIL-science \
actor_rollout_ref.rollout.n=${ROLLOUT_N} \
actor_rollout_ref.model.path=${model_path} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"

    local run_cmd="bash ${REPO_DIR}/training/verl_training.sh ${exp_name} ${CONFIG_NAME} ${data_path} ${args}"
    local setup_cmd="pip install -e ${REPO_DIR} --quiet; export PYTHONPATH=${REPO_DIR}:\$PYTHONPATH"
    local wrapped="srun bash -c '${setup_cmd}; ${run_cmd}'"

    local sbatch_args=(
        sbatch
        --job-name="distil-science"
        --account="${ACCOUNT}"
        --partition="${PARTITION}"
        --nodes=1
        --ntasks-per-node=1
        --gpus-per-node="${GPUS_PER_NODE}"
        --cpus-per-task="${CPUS_PER_TASK}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${REPO_DIR}/../logs/${exp_name}_%j.log"
        --error="${REPO_DIR}/../logs/${exp_name}_%j.err"
        --wrap="${wrapped}"
    )

    if [[ "${DRY_RUN}" == true ]]; then
        echo "----------------------------------------------------------------"
        echo "[dry-run] exp: ${exp_name} | data: ${data_path} | model: ${model_path}"
        echo "${sbatch_args[@]}"
    else
        echo "Submitting: ${exp_name}"
        "${sbatch_args[@]}"
    fi
}

# =============================================================================
# SWEEP
# =============================================================================
mkdir -p "${REPO_DIR}/../logs"

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "${MODEL_PATH}" | tr '/' '-')
    for DATA_PATH in "${DATA_PATHS[@]}"; do
        DOMAIN=$(basename "${DATA_PATH}")
        EXP_NAME="DistIL-science-${DOMAIN}-${MODEL_NAME}"
        submit_job "${EXP_NAME}" "${DATA_PATH}" "${MODEL_PATH}"
    done
done
