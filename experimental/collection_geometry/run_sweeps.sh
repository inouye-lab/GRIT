#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE/../.."
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/grit-matplotlib"
if [ ! -x .venv/bin/python ]; then
    uv sync --locked --group dev
fi
exec .venv/bin/python "$HERE/sweeps.py" "$@"
