#!/bin/bash
set -e

cd "/work/MatiasKokholmAppel#3073/RP_ACT"
source ".venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model"


LOG_DIR='/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/trainer_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
 --num_processes=4 \
 --num_machines=1 \
 --dynamo_backend=no \
 --mixed_precision=bf16 \
 Pipelines/Tsdae_Model/tsdae_model_main.py \
 --data-path /work/RP_act/ACT/Data/clean_combine/message_data.parquet \
 --output-dir-logs "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer.log" \
 --output-dir-model "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT_messagelvl_embed" \
 --epochs 5 \
 --batch-size 32 \
 --lr 1.7739772343958358e-05 \
 --warmup-ratio 0.18892376542532993 \
 --max-grad-norm 1.0 \
 --weight-decay 0.01 \
 2>&1 | tee "$LOGFILE"