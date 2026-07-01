#!/bin/bash
source /work/RP_act/ACT/.venv/bin/activate
cd /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model
source /work/MatiasKokholmAppel#3073/RP_ACT/.venv/bin/activate
LOG_DIR='/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/trainer_$(date +%Y%m%d_%H%M%S).log"
uv run --active accelerate launch \
 --num_processes=3 \
 --mixed_precision=bf16 \
 tsdae_model_main.py \
 2>&1 | tee "$LOGFILE"