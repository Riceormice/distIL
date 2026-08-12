#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 sdpo|sr_opsd 12|16}"
VAL_N="${2:?Usage: $0 sdpo|sr_opsd 12|16}"
case "${METHOD}" in
  sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be sdpo or sr_opsd" >&2; exit 2 ;;
esac
case "${VAL_N}" in
  12|16) ;;
  *) echo "ERROR: VAL_N must be 12 or 16" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_a800_eval.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the exact SDPO or SR-OPSD checkpoint. Use the same checkpoint for N=12 and N=16.}"
CHECKPOINT_DIR="$(readlink -f "${CHECKPOINT_DIR}")"
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "ERROR: checkpoint directory not found: ${CHECKPOINT_DIR}" >&2; exit 2; }

checkpoint_tag="$(basename "${CHECKPOINT_DIR}")"
parent_tag="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
if [[ "${checkpoint_tag}" == "actor" ]]; then
  checkpoint_tag="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
fi
run_tag="${RUN_TAG:-${parent_tag}-${checkpoint_tag}}"
run_tag="$(printf '%s' "${run_tag}" | tr -cs 'A-Za-z0-9._-' '-')"
run_tag="${run_tag%-}"

RUN_ROOT="${OUTPUT_ROOT}/${METHOD}/n${VAL_N}/${run_tag}"
RESULT_DIR="${RUN_ROOT}/results"
MERGED_DIR="${RUN_ROOT}/merged_checkpoint"
STATE_DIR="${RUN_ROOT}/state"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RESULT_DIR}" "${STATE_DIR}" "${LOG_DIR}"

exec 9>"${STATE_DIR}/evaluation.lock"
flock -n 9 || { echo "ERROR: this exact evaluation is already running: ${RUN_ROOT}" >&2; exit 3; }

LAUNCH_LOG="${LOG_DIR}/launcher_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

cleanup() {
  local status=$?
  trap - EXIT
  if [[ "${REMOVE_TEMP_MERGE:-1}" == "1" && "${CREATED_TEMP_MERGE:-0}" == "1" ]]; then
    rm -rf "${MERGED_DIR}"
  fi
  if (( status == 0 )); then
    touch "${STATE_DIR}/complete"
  else
    printf '%s\n' "${status}" >"${STATE_DIR}/last_exit_code"
  fi
  exit "${status}"
}
trap cleanup EXIT

echo "============================================================"
echo "Qwen3-8B math evaluation"
echo "host=$(hostname)"
echo "method=${METHOD}"
echo "samples_per_question=${VAL_N}"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "datasets=AIME24,AIME25,HMMT25,AMC23,Minerva"
echo "thinking=enabled"
echo "temperature=1.0 top_p=1.0 top_k=-1 min_p=0 presence_penalty=0"
echo "max_new_tokens=38912 tensor_parallel=8"
echo "results=${RESULT_DIR}"
echo "online_loggers=disabled"
echo "============================================================"

runtime_preflight
validate_local_datasets

MODEL_DIR=""
LORA_ADAPTER_DIR=""
CREATED_TEMP_MERGE=0

is_adapter_dir() {
  [[ -s "$1/adapter_config.json" ]] &&
    [[ -s "$1/adapter_model.safetensors" || -s "$1/adapter_model.bin" ]]
}

is_full_model_dir() {
  [[ -s "$1/config.json" ]] && {
    compgen -G "$1/*.safetensors" >/dev/null ||
      compgen -G "$1/*.bin" >/dev/null
  }
}

actor_dir=""
if is_adapter_dir "${CHECKPOINT_DIR}"; then
  MODEL_DIR="${BASE_MODEL_DIR}"
  LORA_ADAPTER_DIR="${CHECKPOINT_DIR}"
elif is_full_model_dir "${CHECKPOINT_DIR}"; then
  MODEL_DIR="${CHECKPOINT_DIR}"
  [[ -d "${CHECKPOINT_DIR}/lora_adapter" ]] && is_adapter_dir "${CHECKPOINT_DIR}/lora_adapter" && \
    LORA_ADAPTER_DIR="${CHECKPOINT_DIR}/lora_adapter"
elif [[ -d "${CHECKPOINT_DIR}/actor" ]]; then
  actor_dir="${CHECKPOINT_DIR}/actor"
elif [[ "$(basename "${CHECKPOINT_DIR}")" == "actor" ]]; then
  actor_dir="${CHECKPOINT_DIR}"
else
  echo "ERROR: unsupported checkpoint layout: ${CHECKPOINT_DIR}" >&2
  echo "Expected a LoRA adapter, merged Hugging Face model, global_step_N directory, or actor directory." >&2
  exit 2
fi

if [[ -n "${actor_dir}" ]]; then
  rm -rf "${MERGED_DIR}"
  echo "Merging verl FSDP actor checkpoint: ${actor_dir}"
  "${ENV_DIR}/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${MERGED_DIR}"
  CREATED_TEMP_MERGE=1
  is_full_model_dir "${MERGED_DIR}" || { echo "ERROR: merger did not create a complete model" >&2; exit 2; }
  MODEL_DIR="${MERGED_DIR}"
  [[ -d "${MERGED_DIR}/lora_adapter" ]] && is_adapter_dir "${MERGED_DIR}/lora_adapter" && \
    LORA_ADAPTER_DIR="${MERGED_DIR}/lora_adapter"
fi

export METHOD VAL_N CHECKPOINT_DIR MODEL_DIR LORA_ADAPTER_DIR RESULT_DIR
"${ENV_DIR}/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

checkpoint = Path(os.environ["CHECKPOINT_DIR"])
candidates = sorted(
    path for path in checkpoint.rglob("*")
    if path.is_file() and path.name in {
        "adapter_config.json", "adapter_model.safetensors", "config.json",
        "model.safetensors.index.json", "trainer_state.json"
    }
)
files = []
for path in candidates:
    should_hash = path.stat().st_size < 32 * 1024 * 1024 or path.name.startswith("adapter_model")
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if should_hash else None
    files.append({"path": str(path), "size": path.stat().st_size, "sha256": digest})

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "method": os.environ["METHOD"],
    "samples_per_question": int(os.environ["VAL_N"]),
    "checkpoint_dir": str(checkpoint),
    "model_dir": os.environ["MODEL_DIR"],
    "lora_adapter_dir": os.environ.get("LORA_ADAPTER_DIR") or None,
    "datasets": ["aime24", "aime25", "hmmt25", "amc23", "minerva"],
    "evaluation": {
        "thinking": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 38912,
        "tensor_parallel_size": 8,
    },
    "identity_files": files,
}
path = Path(os.environ["RESULT_DIR"]).parent / "manifest.json"
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote manifest: {path}")
PY

MODEL_DIR="${MODEL_DIR}" \
LORA_ADAPTER_DIR="${LORA_ADAPTER_DIR}" \
MODEL_SIZE=8b \
OUTPUT_DIR="${RESULT_DIR}" \
VAL_N="${VAL_N}" \
TENSOR_PARALLEL_SIZE=8 \
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.88}" \
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" \
PYTHON_BIN="${ENV_DIR}/bin/python" \
bash "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"

"${ENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
rows = []
for dataset in ("aime24", "aime25", "hmmt25", "amc23", "minerva"):
    payload = json.loads((root / f"{dataset}.json").read_text(encoding="utf-8"))
    rows.append({
        "method": os.environ["METHOD"],
        "dataset": dataset,
        "samples": int(os.environ["VAL_N"]),
        "problems": payload["num_problems"],
        "avg": payload["average_at_n_pct"],
        "pass": payload["pass_at_n_pct"],
        "majority": payload["majority_vote_at_n_pct"],
        "format": payload["format_rate"],
    })

path = root.parent / "summary.csv"
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(path.read_text(encoding="utf-8"), end="")
PY

echo "COMPLETE: ${METHOD} Average/Pass/Majority/Format@${VAL_N}"
echo "summary=${RUN_ROOT}/summary.csv"
