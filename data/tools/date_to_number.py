import pandas as pd

def transform_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = (pd.to_datetime(df[date_col]) - pd.to_datetime(df[date_col]).min()).dt.days
    return df

if __name__ == "__main__":
    input_csv = pd.read_csv("points_dataset_thinned.csv") 
    output_csv = transform_date_column(input_csv, "acq_date")
    output_csv.to_csv("processed_fires_dataset.csv", index=False)