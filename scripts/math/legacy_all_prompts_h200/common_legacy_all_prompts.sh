#!/usr/bin/env bash

configure_legacy_all_prompts() {
  local output_root="${1:?output root is required}"
  local marker="${output_root}/evaluation_protocol.env"

  export EVAL_SUBMISSION_MODE=legacy_all_prompts
  export EVAL_PROMPT_BATCH_SIZE=0
  mkdir -p "${output_root}"

  if [[ -f "${marker}" ]]; then
    grep -qx 'protocol=legacy_all_prompts' "${marker}" || {
      echo "ERROR: output root uses a different evaluation protocol: ${marker}" >&2
      return 2
    }
    grep -qx 'prompt_batch_size=0' "${marker}" || {
      echo "ERROR: output root is not an all-prompts run: ${marker}" >&2
      return 2
    }
  elif find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "ERROR: refusing an unmarked non-empty output root: ${output_root}" >&2
    return 2
  else
    printf '%s\n' \
      'protocol=legacy_all_prompts' \
      'prompt_batch_size=0' \
      'samples_per_problem=16' \
      'temperature=1.0' \
      'top_p=1.0' \
      'top_k=-1' \
      'max_new_tokens=38912' \
      'max_model_len=40960' \
      'tensor_parallel_size=8' \
      >"${marker}"
  fi

  echo "Legacy all-prompts evaluation profile: ${marker}"
}
