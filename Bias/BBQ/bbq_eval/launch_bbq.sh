#!/usr/bin/env bash
# Run from any working directory. Paths supplied by the caller stay relative to it.
set -euo pipefail
exec python -m bbq_eval "$@"
