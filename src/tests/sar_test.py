from src.data_pipeline.sentinel_fetcher import SentinelFetcher

fetcher = SentinelFetcher()
res = fetcher.fetch_all_radar_optical(
    lat=52.5708739, lon=-117.9518471, target_date="2021-08-15"
)

print("Optical bands:", list(res.get("bands_t0", {}).keys()))
print("SAR bands content:", res.get("sar_bands"))