#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/work/JosefineMarianneChristensen#0406/RP_ACT"

cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

uv run python -m Pipelines.Derive_Relations_And_Arenas.derive_relations_and_arenas_main