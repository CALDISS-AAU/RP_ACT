#!/bin/bash
source /work/RP_act/ACT/.venv/bin/activate
cd /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model
LOGFILE="trainer_$(date +%Y%m%d_%H%M%S).log"
uv run accelerate launch --num_processes=3 tsdae_model_main.py 2>&1 | tee "LOGFILE"