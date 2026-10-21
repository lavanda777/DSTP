#!/usr/bin/env bash
# Sequential VTAB-19 runs on the GPU explicitly selected by the caller.
set -euo pipefail
dstp_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RUN_DIR:-}${RESUME:-}" ]]; then
    echo "Use individual scripts for RUN_DIR/RESUME; all runs require distinct outputs." >&2
    exit 2
fi
for dstp_arg in "$@"; do
    case "$dstp_arg" in --output|--output=*|--resume|--resume=*|--dataset|--dataset=*)
        echo "run_all does not accept shared output/resume/dataset overrides." >&2; exit 2;;
    esac
done
for dstp_dataset in caltech101 cifar100 dtd flowers102 pets sun397 svhn \
    patch_camelyon eurosat resisc45 diabetic_retinopathy \
    clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele; do
    bash "$dstp_script_dir/$dstp_dataset.sh" "$@"
done
