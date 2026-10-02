uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data.parquet \
    --split-seed 1849 \
    --name msg_seed_1849

uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet \
    --split-seed 1849 \
    --name msg_no_advisor_seed_1849

uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/conversation_data.parquet \
    --split-seed 1849 \
    --name conversation_seed_1849

uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/conversation_data_without_advisors.parquet \
    --split-seed 1849 \
    --name conversation_no_advisor_seed_1849

uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet \
    --split-seed 1849 \
    --name collapsed_seed_1849

uv run python -m Pipelines.Data_Prep.data_prep_main \
    --data-path /work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet \
    --split-seed 1849 \
    --name collapsed_no_advisor_seed_1849
