#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDPO_DIR="${REPO_ROOT}/SDPO"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-20}"
NUM_GPUS="${NUM_GPUS:-8}"
SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}"

case "${MODEL_SIZE}" in
  4b)
    MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B-Instruct-2507}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    ;;
  8b)
    MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b, got ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

DATA_JSONL="${DATA_JSONL:-${REPO_ROOT}/OPSD/data/math/train.jsonl}"
DATA_DIR="${DATA_DIR:-/media/vlm-ckp-fileset/ylong/sdpo_math_data/distil_math}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_verl_math}"
RUN_NAME="${RUN_NAME:-sr-opsd-${MODEL_SIZE}-seed${SEED}-rho0.95-refw${SELF_REFERENCE_WEIGHT}-sync0-lr5e-6-tok16384-steps${TOTAL_STEPS}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs/${RUN_NAME}}"

mkdir -p "${DATA_DIR}" "${RUN_DIR}" "${LOG_DIR}"
test -x "${PYTHON_BIN}"
test -f "${MODEL_PATH}/config.json"
test -f "${DATA_JSONL}"

export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"
export USER="${USER:-root}"
export TASK="distil_math"
export EXPERIMENT="${RUN_NAME}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export VERL_FILE_LOGGER_PATH="${LOG_DIR}/metrics.jsonl"
export NUM_GPUS SELF_REFERENCE_WEIGHT

if [[ ! -s "${DATA_DIR}/train.parquet" || ! -s "${DATA_DIR}/test.parquet" || "${DATA_JSONL}" -nt "${DATA_DIR}/train.parquet" ]]; then
  "${PYTHON_BIN}" "${SDPO_DIR}/examples/data_preprocess/distil_math_jsonl.py" \
    --input "${DATA_JSONL}" \
    --output-dir "${DATA_DIR}"
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import ray
import vllm
from verl.trainer.ppo.sr_opsd_loss import forward_renyi_divergence, geometric_target_log_probs
from verl.workers.config.actor import SelfDistillationConfig

assert torch.cuda.device_count() == int(__import__("os").environ.get("NUM_GPUS", "8"))
print("torch", torch.__version__)
print("ray", ray.__version__)
print("vllm", vllm.__version__)
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))

reference_teacher_weight = float(__import__("os").environ["SELF_REFERENCE_WEIGHT"])
config = SelfDistillationConfig(
    divergence="renyi_forward",
    rho=0.95,
    reference_policy=True,
    reference_teacher_weight=reference_teacher_weight,
    reference_sync_steps=0,
    teacher_regularization="ema",
)
student = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
teacher = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
reference = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
target = geometric_target_log_probs(teacher, reference, config.reference_teacher_weight)
loss = forward_renyi_divergence(target, student, config.rho).mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
print("SR-OPSD numerical smoke test: PASS", float(loss))
PY

cd "${SDPO_DIR}"

ARGS=(
  "data.train_files=${DATA_DIR}/train.parquet"
  "data.val_files=${DATA_DIR}/test.parquet"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=2048"
  "data.max_response_length=16384"
  "data.seed=${SEED}"
  "data.shuffle=True"
  'data.apply_chat_template_kwargs={enable_thinking: true}'
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.lora_rank=0"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.calculate_entropy=True"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.optim.lr=5e-6"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=linear"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.optim.weight_decay=0"
  "actor_rollout_ref.actor.optim.clip_grad=0.1"
  "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  "actor_rollout_ref.actor.self_distillation.alpha=0.25"
  "actor_rollout_ref.actor.self_distillation.divergence=renyi_forward"
  "actor_rollout_ref.actor.self_distillation.rho=0.95"
  "actor_rollout_ref.actor.self_distillation.reference_policy=True"
  "actor_rollout_ref.actor.self_distillation.reference_teacher_weight=${SELF_REFERENCE_WEIGHT}"
  "actor_rollout_ref.actor.self_distillation.reference_sync_steps=0"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05"
  "actor_rollout_ref.actor.self_distillation.max_reprompt_len=20480"
  "actor_rollout_ref.actor.self_distillation.reprompt_truncation=right"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
  "actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=False"
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True"
  "actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True"
  "actor_rollout_ref.actor.self_distillation.is_clip=null"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.temperature=0.7"
  "actor_rollout_ref.rollout.top_p=0.95"
  "actor_rollout_ref.rollout.top_k=20"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
  "actor_rollout_ref.rollout.max_model_len=38912"
  "actor_rollout_ref.rollout.max_num_batched_tokens=38912"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=False"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=2.0"
  "custom_reward_function.path=${SDPO_DIR}/verl/utils/reward_score/feedback/__init__.py"
  "reward_model.use_reward_loop=False"
  "trainer.project_name=SR-OPSD-Math"
  "trainer.group_name=sr_opsd_math"
  "trainer.experiment_name=${RUN_NAME}"
  'trainer.logger=[console,file]'
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.nnodes=1"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=-1"
  "trainer.val_before_train=False"
  "trainer.resume_mode=auto"
  "trainer.max_actor_ckpt_to_keep=null"
  "trainer.default_local_dir=${RUN_DIR}"
  "trainer.use_legacy_worker_impl=enable"
)

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "PARAMETERS=rho0.95 refw${SELF_REFERENCE_WEIGHT} sync0 ema0.05 lr5e-6 linear warmup0 tok16384"

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo "${ARGS[@]}"
