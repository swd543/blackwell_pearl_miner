#!/usr/bin/env bash
# Local power policy helper. All GPU power changes are limited to these two
# explicit values and performed through the configured sudoers rule for NVML.
set -euo pipefail

case "${1:-}" in
  250|300) limit=$1 ;;
  *) echo "usage: $0 {250|300}" >&2; exit 2 ;;
esac

current=$(/usr/bin/nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)
if [[ "$current" == "$limit" ]]; then
  exit 0
fi
if sudo -n /usr/bin/nvidia-smi -pl "$limit" >/dev/null 2>&1; then
  printf 'GPU power limit: %sW -> %sW\n' "${current:-unknown}" "$limit" >&2
else
  echo "failed to set GPU power limit to ${limit}W" >&2
  exit 1
fi
