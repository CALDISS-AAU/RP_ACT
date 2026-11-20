#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/work/JosefineMarianneChristensen#0406/RP_ACT"

cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "This bash script rocks"
uv run python -m Pipelines.Kwic.kwic_main
echo "But now it's over"