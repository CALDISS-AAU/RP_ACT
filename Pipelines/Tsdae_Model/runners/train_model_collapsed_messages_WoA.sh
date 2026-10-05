#!/bin/bash
set -e

cd "/work/JosefineMarianneChristensen#0406/RP_ACT"
source ".venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:./Pipelines/Tsdae_Model"


LOG_DIR='./Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/collapsed_messages_WoA_trainer_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
 --num_processes=4 \
 --num_machines=1 \
 --dynamo_backend=no \
 --mixed_precision=bf16 \
 Pipelines/Tsdae_Model/tsdae_model_main.py \
 --dataset-dir /work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_no_advisor_seed_1849 \
 --output-dir-logs "./Pipelines/Tsdae_Model/Logs/trainer_collapsed_WoA.log" \
 --output-dir-model "/work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed_WoA" \
 --epochs 6 \
 --batch-size 4 \
 --lr 1.728068498248166e-05 \
 --warmup-ratio 0.018328394014304483 \
 --max-grad-norm 1.0 \
 --weight-decay 0.07786111579093544 \
 2>&1 | tee "$LOGFILE"