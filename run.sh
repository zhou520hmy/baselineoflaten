#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_ROOT"
export LATEN_BUNDLE_ROOT="$BUNDLE_ROOT"
export LATEN_STORE="${LATEN_STORE:-$BUNDLE_ROOT/storage}"
mkdir -p "$LATEN_STORE" "$BUNDLE_ROOT/logs"
LATEN_STORE="$(cd "$LATEN_STORE" && pwd)"; export LATEN_STORE
export HF_HOME="$LATEN_STORE/cache/huggingface"
export HF_DATASETS_CACHE="$LATEN_STORE/cache/datasets"
export XDG_CACHE_HOME="$LATEN_STORE/cache/runtime"
export PIP_CACHE_DIR="$LATEN_STORE/cache/pip"
export TORCH_HOME="$LATEN_STORE/cache/torch"
export TRITON_CACHE_DIR="$LATEN_STORE/cache/triton"
export TMPDIR="$LATEN_STORE/tmp"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
ACTION="${1:-all}"; if (($#)); then shift; fi
if [[ "$ACTION" == validate ]]; then
  exec "${PYTHON_BIN:-python3}" -c "import unittest,sys; s=unittest.defaultTestLoader.discover('tests',pattern='test_[sd]*.py'); assert s.countTestCases()>=33, 'Missing tests'; sys.exit(not unittest.TextTestRunner(verbosity=2).run(s).wasSuccessful())"
fi
if [[ "$ACTION" == status || "$ACTION" == report ]]; then
  exec "${PYTHON_BIN:-python3}" -m suite.cli "$ACTION" "$@"
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if [[ "$ACTION" == all || "$ACTION" == bootstrap ]]; then
  "${PYTHON_BIN:-python3}" -m suite.cli preflight-host "$@"
  if [[ ! -x .venv/bin/python ]]; then "${PYTHON_BIN:-python3}" -m venv .venv; fi
  .venv/bin/python -m pip install --upgrade 'pip==25.1.1'
  .venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip check
  if [[ "$ACTION" == bootstrap ]]; then exit 0; fi
fi
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash run.sh bootstrap first.' >&2; exit 2; fi
exec .venv/bin/python -m suite.cli "$ACTION" "$@"
