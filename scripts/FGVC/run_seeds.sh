#!/usr/bin/env bash
# One dataset or all five, sequentially over the shared seed list. No implicit GPU choice.
# DRY_RUN=1 prints commands only; --print-config runs config parsing only.
set -euo pipefail
dstp_dataset="${1:?Usage: run_seeds.sh DATASET|all [DSTP CLI overrides]}"
shift
case "$dstp_dataset" in cub|nabirds|flowers|dogs|cars) dstp_target="$dstp_dataset.sh" ;; all) dstp_target="run_all.sh" ;; *) echo "Unknown dataset" >&2; exit 2 ;; esac
dstp_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dstp_project_root="$(cd -- "$dstp_script_dir/../.." && pwd)"
dstp_python="${PYTHON_BIN:-/home/pc/anaconda3/envs/GPS/bin/python}"
dstp_options=(--seed-file "${SEED_FILE:-$dstp_project_root/configs/shared_seeds.json}")
if [[ -n "${SEEDS:-}" ]]; then
    read -r -a dstp_seeds <<< "$SEEDS"
    dstp_options+=(--seeds "${dstp_seeds[@]}")
fi
[[ "${DRY_RUN:-0}" != 1 ]] || dstp_options+=(--dry-run)
cd -- "$dstp_project_root"
exec "$dstp_python" -m dstp.seed_sweep "${dstp_options[@]}" -- bash "$dstp_script_dir/$dstp_target" "$@"
