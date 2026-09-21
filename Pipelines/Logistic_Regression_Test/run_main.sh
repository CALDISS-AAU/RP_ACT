PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Logistic_Regression_Test/logistic_regression_test_main.py \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet \
    --embedding-path /work/RP_act/ACT/Data/embeddings/collapsed_without_advisors.npy \
    --C 1.0 \
    --class-weight balanced \
    --filname collapsed_without_advisors

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Logistic_Regression_Test/logistic_regression_test_main.py \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet \
    --embedding-path /work/RP_act/ACT/Data/embeddings/msg_lvl_without_advisors_embeddings.npy \
    --C 1.0 \
    --class-weight balanced \
    --filname message_without_advisors

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Logistic_Regression_Test/logistic_regression_test_main.py \
    --data-path /work/RP_act/ACT/Data/embeddings/bge_m3_baseline_embeddings.npy \
    --embedding-path /work/RP_act/ACT/Data/embeddings/bge_m3_baseline_embeddings.npy \
    --C 1.0 \
    --class-weight balanced \
    --filname bge_m3_message_without_advisors