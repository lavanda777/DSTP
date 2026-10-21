#!/usr/bin/env bash
# Shared launcher only. Dataset-specific numerical defaults live in configs/fgvc/.
set -euo pipefail
dstp_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

dstp_run() {
    local dstp_dataset="$1"
    shift
    local dstp_python="${PYTHON_BIN:-/home/pc/anaconda3/envs/GPS/bin/python}"
    local -a dstp_overrides=()
    local dstp_key dstp_flag
    for dstp_key in LR HEAD_LR BETA1 BETA2 EPOCHS \
        COOLDOWN_EPOCHS WARMUP_EPOCHS WARMUP_LR MIN_LR LR_INTERVAL BATCH_SIZE VALIDATION_BATCH_SIZE WORKERS SEED \
        AMP DEVICE DROP_PATH AUGMENTATION MIXUP CUTMIX MIXUP_PROB MIXUP_SWITCH_PROB MIXUP_MODE \
        SMOOTHING EMA_DECAY WEIGHT_DECAY \
        DATA_ROOT SPLIT_ROOT PRETRAINED CHECKPOINT_EVERY LOG_EVERY STOP_AFTER_STEPS; do
        if [[ -n "${!dstp_key-}" ]]; then
            dstp_flag="${dstp_key,,}"
            dstp_overrides+=("--${dstp_flag//_/-}" "${!dstp_key}")
        fi
    done
    [[ -z "${RUN_DIR:-}" ]] || dstp_overrides+=(--output "$RUN_DIR")
    [[ -z "${RESUME:-}" ]] || dstp_overrides+=(--resume "$RESUME")
    cd -- "$dstp_project_root"
    export PYTHONUNBUFFERED=1
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    # Caller chooses physical GPU via CUDA_VISIBLE_DEVICES. CLI overrides env.
    exec "$dstp_python" -m dstp.train --dataset "$dstp_dataset" "${dstp_overrides[@]}" "$@"
}
