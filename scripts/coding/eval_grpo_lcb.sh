#!/bin/bash
# scripts/coding/eval_grpo.sh
# Evaluate a GRPO checkpoint on LCBv6.
#
# Usage:
#   CHECKPOINT=GRPO-off_policy-coding-lcbv6-Qwen-Qwen3-8B \
#   CHECKPOINT_STEP=80 \
#   REPO_DIR=/path/to/distil/SDPO \
#   bash scripts/coding/eval_grpo.sh

set -euo pipefail

# =============================================================================
# CLUSTER CONFIGURATION — edit these for your cluster
# =============================================================================
ACCOUNT="${ACCOUNT:-your_account}"
PARTITION="${PARTITION:-gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TIME="${TIME:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-256000}"
REPO_DIR="${REPO_DIR:-/path/to/distil/SDPO}"

# =============================================================================
# EVAL CONFIGURATION
# =============================================================================
CHECKPOINT="${CHECKPOINT:-GRPO-off_policy-coding-lcbv6-Qwen-Qwen3-8B}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-80}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# GRPO uses temperature=0.6 for eval (same as training val_kwargs)
VAL_N=16
VAL_TEMPERATURE=0.2
VAL_TOP_P=0.95
VAL_TOP_K=20

# =============================================================================
# JOB SUBMISSION
# =============================================================================
CKPT_DIR=$(grep "ckpt_dir" "${REPO_DIR}/verl/trainer/config/user.yaml" | awk '{print $2}')
echo "Setting checkpoint step to ${CHECKPOINT_STEP} for: ${CHECKPOINT}"
echo "${CHECKPOINT_STEP}" > "${CKPT_DIR}/${CHECKPOINT}/latest_checkpointed_iteration.txt"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP="${CHECKPOINT}"
LOG="${REPO_DIR}/../logs/eval_grpo_lcb_${TIMESTAMP}.log"
SBATCH_SCRIPT="${REPO_DIR}/../logs/.sbatch_eval_grpo_lcb_${TIMESTAMP}.sh"

mkdir -p "${REPO_DIR}/../logs"

cat > "${SBATCH_SCRIPT}" << SBEOF
#!/bin/bash
#SBATCH --job-name=eval-grpo-lcb
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

echo "===== JOB START ====="
date
hostname

pip install -e ${REPO_DIR} --quiet
export PYTHONPATH=${REPO_DIR}:\$PYTHONPATH
export HF_HOME=\${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_OFFLINE=1

bash ${REPO_DIR}/training/verl_training.sh '${EXP}' ppo_trainer 'datasets/lcb_v6' \
  vars.dir=${REPO_DIR} \
  data.train_batch_size=32 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=${GPUS_PER_NODE}

echo "===== JOB END ====="
date
SBEOF

JOB=$(sbatch "${SBATCH_SCRIPT}")
echo "Submitted: ${JOB}"
echo "Logs: ${LOG}"
