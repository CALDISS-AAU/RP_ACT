"""Main script for the Tsdae_Model pipeline."""
# Internal imports 
from Messagelvel_Model import create_strata_col, split_data, build_sentence_transformer, train_tsdae, INPUT_DIR_DATA

def main():
    model = build_sentence_transformer(model_name)
    dataset = read_parquet(INPUT_DIR_DATA)
    dataset = create_strata_col(dataset)
    train_df, val_df, test_df = split_data(dataset)
    model = train_tsdae(model, train_df, val_df)


if __name__ == "__main__":
    main()

