#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Same store/config preserves known completed training-free evaluation cells.
# Pass --family 4b or --family 8b to select a model explicitly.
exec bash "$BUNDLE_ROOT/run.sh" resume "$@"
