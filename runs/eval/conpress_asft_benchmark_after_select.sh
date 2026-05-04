#!/usr/bin/env bash
set -euo pipefail

echo "archived: use runs/archive/20260504/conpress_asft_benchmark_after_select.sh" >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/runs/archive/20260504/conpress_asft_benchmark_after_select.sh" "$@"
