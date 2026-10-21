#!/usr/bin/env bash
# SEEDS="42 43 44"; DRY_RUN=1 previews without training.
set -euo pipefail
dstp_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dstp_dataset="${1:?Usage: run_seeds.sh DATASET [training flags]}"
shift
case "$dstp_dataset" in caltech101|cifar100|dtd|flowers102|pets|sun397|svhn|patch_camelyon|eurosat|resisc45|diabetic_retinopathy|clevr_count|clevr_dist|dmlab|kitti|dsprites_loc|dsprites_ori|smallnorb_azi|smallnorb_ele) ;;
    *) echo "Unknown VTAB dataset: $dstp_dataset" >&2; exit 2;;
esac
cd -- "$dstp_script_dir/../.."
read -r -a dstp_seeds <<< "${SEEDS:-42 43 44}"
dstp_dry=()
[[ "${DRY_RUN:-0}" != 1 ]] || dstp_dry+=(--dry-run)
exec "${PYTHON_BIN:-/home/pc/anaconda3/envs/GPS/bin/python}" -m dstp.seed_sweep \
    "${dstp_dry[@]}" --seeds "${dstp_seeds[@]}" -- bash "$dstp_script_dir/$dstp_dataset.sh" "$@"
