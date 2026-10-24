import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import List, Optional, Tuple
import pandas as pd

from src.config import RAW_DATA_DIR
from src.data_pipeline.fire_date import Date


def process_single_date(
    target_date: date,
    country_code: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    cluster_radius_km: float = 15.0,
    lookback_days: int = 15,
    n_random_negatives: int = 2,
    direction_north: float = 0.0,
    direction_east: float = 0.0,
    intensity: float = 0.0,
) -> Optional[pd.DataFrame]:
    """process single date and return combined daily dataframe of verified fire centroids and negatives"""
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        # pass bounds directly so ESP and PRT share the 5d firms cache and clip overseas islands
        date_obj = Date(target_date, country_code=country_code, bounds=bounds)
        date_obj.generate_fires()

        if date_obj.df_area.empty:
            return None

        # aggregate verified points into fire centroids and filter ongoing burns via lookback
        date_obj.filter_fires(
            cluster_radius_km=cluster_radius_km,
            lookback_days=lookback_days,
        )
        if date_obj.df_area_filtered.empty:
            return None

        n_fires = len(date_obj.df_area_filtered)

        # generate 1 hard negative per confirmed fire centroid
        date_obj.generate_hard_negatives(buffer_days=lookback_days)

        # scale random negatives with found fires
        n_rand = min(n_random_negatives, n_fires) if n_random_negatives > 0 else 0
        if n_rand > 0:
            date_obj.generate_random_negatives(
                n_points=n_rand,
                buffer_days=lookback_days,
                direction_north=direction_north,
                direction_east=direction_east,
                intensity=intensity,
            )

        df_day = date_obj.get_combined_dataset()
        if not df_day.empty:
            n_neg = len(df_day) - n_fires
            print(f"processed {date_str} ({country_code}): {len(df_day)} points ({n_fires} fires, {n_neg} negatives)")
            return df_day
    except Exception as e:
        print(f"error processing date {date_str} ({country_code}): {e}")
    return None

def _build_date_tasks(
    start_year: int,
    end_year: int,
    months: List[int],
    days: Optional[List[int]],
    country_codes: List[str],
) -> List[Tuple[date, str]]:
    """build valid calendar (date, country_code) tasks; if days is None, include all days in each month"""
    tasks = []
    for year in range(start_year, end_year + 1):
        for month in months:
            max_day = calendar.monthrange(year, month)[1]
            target_days = range(1, max_day + 1) if days is None else [d for d in days if 1 <= d <= max_day]
            for day in target_days:
                for code in country_codes:
                    tasks.append((date(year, month, day), code))
    return tasks


def generate_master_dataset(
    start_year: int = 2021,
    end_year: int = 2024,
    months: List[int] = [6, 7, 8, 9],
    days: Optional[List[int]] = None,
    country_codes: List[str] = ["CAN"],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    cluster_radius_km: float = 3.0,
    lookback_days: int = 15,
    n_random_negatives: int = 2,
    direction_north: float = 0.0,
    direction_east: float = 0.0,
    intensity: float = 0.0,
    output_filename: str = "master_points_dataset.csv",
    max_workers: int = 4,
) -> pd.DataFrame:
    """batch dataset generator using parallel threads, 10-day firms caching, and effis polygon pre-checks"""
    target_tasks = _build_date_tasks(
        start_year=start_year,
        end_year=end_year,
        months=months,
        days=days,
        country_codes=country_codes,
    )

    output_filepath = RAW_DATA_DIR / output_filename

    target_desc = f"bounds {bounds}" if bounds else f"countries {country_codes}"
    mode_desc = "all calendar days" if days is None else f"days {days}"
    print(
        f"starting parallel generation for {target_desc} [{mode_desc}] "
        f"({len(target_tasks)} tasks, {max_workers} workers)..."
    )

    all_daily_datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_date,
                d,
                code,
                bounds,
                cluster_radius_km,
                lookback_days,
                n_random_negatives,
                direction_north,
                direction_east,
                intensity,
            )
            for d, code in target_tasks
        ]

        for future in as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                all_daily_datasets.append(res)

    if all_daily_datasets:
        master_df = pd.concat(all_daily_datasets, ignore_index=True)
        master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        master_df.to_csv(output_filepath, index=False)

        total_fires = int((master_df["is_fire"] == 1).sum())
        total_negatives = int((master_df["is_fire"] == 0).sum())

        print("-" * 40)
        print("dataset generation complete")
        print(f"active country-days with fresh fires: {len(all_daily_datasets)}")
        print(f"total rows: {len(master_df)} (fires: {total_fires}, negatives: {total_negatives})")
        print(f"saved to: {output_filepath}")
        print("-" * 40)

        return master_df
    else:
        print("generation finished with no data collected")
        return pd.DataFrame()


if __name__ == "__main__":
    # min_lat, min_lon, max_lat, max_lon
    iberia_bounds = (35.9, -9.5, 43.9, 3.4)

    generate_master_dataset(
        start_year=2015,
        end_year=2017,
        months=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        days=None,  # scan all calendar days (effis pre-check + 10d cache makes this fast)
        country_codes=["ESP", "PRT"],
        bounds=iberia_bounds,
        cluster_radius_km=20.0,
        lookback_days=15,
        n_random_negatives=2,
        direction_north=1.0,
        direction_east=0.3,
        intensity=0.1,
        output_filename="master_points_dataset.csv",
        max_workers=4,
    )