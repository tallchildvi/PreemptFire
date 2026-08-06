import time
import pandas as pd
from datetime import date
from fire_date import Date

def generate_master_dataset(
    start_year: int = 2021,
    end_year: int = 2024,
    months: list = [4, 5, 6, 7, 8, 9, 10, 11],
    days: list = [1, 15],
    country_code: str = "CAN",
    output_filename: str = "master_points_dataset.csv"
) -> pd.DataFrame:
    """batch dataset generator across multiple dates"""
    all_daily_datasets = []
    processed_dates_count = 0

    print(f"starting dataset generation for {country_code} ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        for month in months:
            for day in days:
                target_date = date(year, month, day)
                date_str = target_date.strftime("%Y-%m-%d")

                try:
                    date_obj = Date(target_date, country_code=country_code)
                    date_obj.generate_fires()

                    if date_obj.df_area.empty:
                        continue

                    date_obj.filter_fires(epsilon=0.012)
                    if date_obj.df_area_filtered.empty:
                        continue

                    date_obj.generate_hard_negatives(buffer_days=10)
                    date_obj.generate_random_negatives(n_points=5, buffer_days=10)

                    df_day = date_obj.get_combined_dataset()

                    if not df_day.empty:
                        all_daily_datasets.append(df_day)
                        processed_dates_count += 1
                        print(f"processed {date_str}: {len(df_day)} points")

                except Exception as e:
                    print(f"error processing date {date_str}: {e}")

                time.sleep(1.0)

    if all_daily_datasets:
        master_df = pd.concat(all_daily_datasets, ignore_index=True)
        master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        master_df.to_csv(output_filename, index=False)

        print("-" * 40)
        print("dataset generation complete")
        print(f"processed dates: {processed_dates_count}")
        print(f"total rows: {len(master_df)}")
        print(f"saved to: {output_filename}")
        print("-" * 40)

        return master_df
    else:
        print("generation finished with no data collected")
        return pd.DataFrame()


if __name__ == "__main__":
    generate_master_dataset(
        start_year=2021,
        end_year=2024,
        months=[4, 5, 6, 7, 8, 9, 10, 11],
        days=[1, 15],
        country_code="CAN",
        output_filename="master_points_dataset.csv"
    )