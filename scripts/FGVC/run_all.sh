#!/usr/bin/env bash
# Sequential on the visible GPU. No background jobs or implicit multi-GPU use.
set -euo pipefail
dstp_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RUN_DIR:-}${RESUME:-}" ]]; then
    echo "run_all requires separate automatically named outputs; run individual scripts for resume." >&2
    exit 2
fi
for dstp_arg in "$@"; do
    case "$dstp_arg" in --output|--output=*|--resume|--resume=*|--dataset|--dataset=*)
        echo "run_all does not accept shared output/resume/dataset overrides." >&2; exit 2;;
    esac
done
for dstp_dataset in cub nabirds flowers dogs cars; do
    bash "$dstp_script_dir/$dstp_dataset.sh" "$@"
done
