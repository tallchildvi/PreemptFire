from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import os
from typing import Optional, Tuple
import pandas as pd

from src.config import RAW_DATA_DIR
from src.data_pipeline.fire_date import Date


def process_single_date(
    target_date: date,
    country_code: Optional[str] = "CAN",
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[pd.DataFrame]:
    """process single date and return combined daily dataframe with optional bounds filter."""
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        date_obj = Date(target_date, country_code=country_code)
        date_obj.generate_fires()

        if date_obj.df_area.empty:
            return None

        # optional spatial filtering by bounding box (min_lat, min_lon, max_lat, max_lon)
        if bounds is not None:
            min_lat, min_lon, max_lat, max_lon = bounds
            lat_col = "latitude" if "latitude" in date_obj.df_area.columns else "lat"
            lon_col = "longitude" if "longitude" in date_obj.df_area.columns else "lon"

            date_obj.df_area = date_obj.df_area[
                (date_obj.df_area[lat_col] >= min_lat)
                & (date_obj.df_area[lat_col] <= max_lat)
                & (date_obj.df_area[lon_col] >= min_lon)
                & (date_obj.df_area[lon_col] <= max_lon)
            ].copy()

            if date_obj.df_area.empty:
                return None

        date_obj.filter_fires(epsilon=0.012)
        if date_obj.df_area_filtered.empty:
            return None

        date_obj.generate_hard_negatives(buffer_days=15)
        date_obj.generate_random_negatives(n_points=5, buffer_days=15)

        df_day = date_obj.get_combined_dataset()
        if not df_day.empty:
            print(f"processed {date_str}: {len(df_day)} points")
            return df_day
    except Exception as e:
        print(f"error processing date {date_str}: {e}")
    return None


def generate_master_dataset(
    start_year: int = 2021,
    end_year: int = 2024,
    months: list = [6, 7, 8, 9],
    days: list = [1, 15],
    country_code: Optional[str] = "CAN",
    bounds: Optional[Tuple[float, float, float, float]] = None,
    output_filename: str = "master_points_dataset.csv",
    max_workers: int = 4,
) -> pd.DataFrame:
    """batch dataset generator using parallel threads with country and bounding box support."""
    target_dates = [
        date(year, month, day)
        for year in range(start_year, end_year + 1)
        for month in months
        for day in days
    ]

    output_filepath = RAW_DATA_DIR / output_filename

    target_desc = f"bounds {bounds}" if bounds else f"country {country_code}"
    print(f"starting parallel generation for {target_desc} ({len(target_dates)} dates, {max_workers} workers)...")

    all_daily_datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_date, d, country_code, bounds)
            for d in target_dates
        ]

        for future in as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                all_daily_datasets.append(res)

    if all_daily_datasets:
        master_df = pd.concat(all_daily_datasets, ignore_index=True)
        master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        master_df.to_csv(output_filepath, index=False)

        print("-" * 40)
        print("dataset generation complete")
        print(f"processed dates with data: {len(all_daily_datasets)}")
        print(f"total rows: {len(master_df)}")
        print(f"saved to: {output_filepath}")
        print("-" * 40)

        return master_df
    else:
        print("generation finished with no data collected")
        return pd.DataFrame()


if __name__ == "__main__":
    # British Columbia bounding box
    bc_bounds = (49.0, -139.0, 60.0, -114.0)

    generate_master_dataset(
        start_year=2020,
        end_year=2025,
        months=[4, 5, 6, 7, 8, 9, 10, 11], 
        days=[1, 15],
        country_code="CAN",
        bounds=bc_bounds,
        output_filename="master_points_dataset.csv",
        max_workers=4,
    )