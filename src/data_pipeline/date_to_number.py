import pandas as pd
from src.config import INTERIM_DATA_DIR


def transform_date_column(df: pd.DataFrame, date_col: str = "acq_date") -> pd.DataFrame:
    """transform date column to numeric day offset from the earliest date"""
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df[date_col] = (dates - dates.min()).dt.days
    return df


def main():
    """read thinned points, convert dates to numeric offset, and save result"""
    input_file = INTERIM_DATA_DIR / "points_dataset_thinned.csv"
    output_file = INTERIM_DATA_DIR / "processed_fires_dataset.csv"

    if not input_file.exists():
        print(f"error: input file not found at {input_file}")
        return

    print(f"loading dataset from: {input_file}")
    df_input = pd.read_csv(input_file)

    df_output = transform_date_column(df_input, date_col="acq_date")

    df_output.to_csv(output_file, index=False)
    print(f"successfully saved processed dataset to: {output_file}")


if __name__ == "__main__":
    main()