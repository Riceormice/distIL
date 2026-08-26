#!/usr/bin/env bash

lock_protocol_file() {
  local target="${1:?protocol target is required}"
  local candidate="${2:?protocol candidate is required}"

  if [[ -f "${target}" ]]; then
    if ! cmp -s "${target}" "${candidate}"; then
      echo "ERROR: refusing to resume with a different experiment protocol: ${target}" >&2
      diff -u "${target}" "${candidate}" >&2 || true
      rm -f -- "${candidate}"
      return 2
    fi
    rm -f -- "${candidate}"
    echo "Experiment protocol lock: PASS (${target})"
    return 0
  fi

  mv -- "${candidate}" "${target}"
  chmod 0444 "${target}"
  echo "Experiment protocol locked: ${target}"
}
