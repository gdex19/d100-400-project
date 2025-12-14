from final_project.data import read_data
from final_project.preprocessing import (
    add_features_responder,
    winsorize_predictors,
    write_data,
)


def prepare_data() -> None:
    """
    Prepare Bitcoin data and save
    to disk.
    """
    df_raw = read_data("btc")
    df = add_features_responder(df_raw)

    write_data(df, "clean_data.pq")

    df_clipped = winsorize_predictors(df)
    write_data(df_clipped, "clean_data_clipped.pq")


if __name__ == "__main__":
    prepare_data()
