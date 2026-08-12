#!/usr/bin/env bash
set -Eeuo pipefail

MACHINE="${1:?Usage: $0 1|2|3|4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${MACHINE}" in
  1) ENTRY=a800_machine1_grpo.sh ;;
  2) ENTRY=a800_machine2_sdpo.sh ;;
  3) ENTRY=a800_machine3_opsd.sh ;;
  4) ENTRY=a800_machine4_sr_opsd.sh ;;
  *) echo "ERROR: machine must be 1, 2, 3, or 4" >&2; exit 2 ;;
esac
exec bash "${SCRIPT_DIR}/${ENTRY}"
