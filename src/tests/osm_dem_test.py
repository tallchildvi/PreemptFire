import time
import matplotlib.pyplot as plt
from src.data_pipeline.osm_dem_fetcher import SpatialFeatureFetcher

TEST_LAT = 52.5708739
TEST_LON = -117.9518471
TEST_DATE = "2021-08-15"
INCLUDE_TRAILS = True


def test_fetcher_roads():
    fetcher = SpatialFeatureFetcher()

    t0 = time.perf_counter()
    roads_raster = fetcher.fetch_road_raster(
        lat=TEST_LAT,
        lon=TEST_LON,
        target_date=TEST_DATE,
        include_trails=INCLUDE_TRAILS,
    )
    elapsed = time.perf_counter() - t0

    road_pixels = int((roads_raster == 1).sum())
    total_pixels = roads_raster.size
    coverage_pct = (road_pixels / total_pixels) * 100.0

    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Shape: {roads_raster.shape}")
    print(f"Active pixels: {road_pixels} / {total_pixels} ({coverage_pct:.2f}%)")

    title_suffix = "Roads + Trails" if INCLUDE_TRAILS else "Roads Only"

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(roads_raster, cmap="inferno", origin="upper")
    ax.set_title(f"OSM Mask ({title_suffix})\n[{TEST_LAT}, {TEST_LON}] ({TEST_DATE})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_fetcher_roads()