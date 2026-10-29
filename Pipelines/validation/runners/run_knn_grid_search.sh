PYTHONPATH="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation:/work/MatiasKokholmAppel#3073/RP_ACT${PYTHONPATH:+:$PYTHONPATH}" \
uv run python \
    -m cuml.accel --profile \
    "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/knn_main.py" \
    --split-dir /work/RP_act/ACT/Data/splits/message_data/seed_1849 \
    --seed 1849 \
    --filename knn_message_data_seed_1849 \
    --grid-search-only