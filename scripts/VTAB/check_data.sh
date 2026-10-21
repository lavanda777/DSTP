#!/usr/bin/env bash
set -euo pipefail
dstp_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$dstp_root"
exec "${PYTHON_BIN:-/home/pc/anaconda3/envs/GPS/bin/python}" -m dstp.check_data --suite vtab "$@"
