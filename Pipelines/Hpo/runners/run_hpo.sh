#!/bin/bash
source /work/MatiasKokholmAppel#3073/RP_ACT/.venv/bin/activate

cd /work/MatiasKokholmAppel#3073/RP_ACT || exit 1
export PYTHONPATH="/work/MatiasKokholmAppel#3073/RP_ACT:$PYTHONPATH"

LOG_DIR="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Hpo/Logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/hpo_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  Pipelines/Hpo/hpo_main.py \
  2>&1 | tee "$LOGFILE"