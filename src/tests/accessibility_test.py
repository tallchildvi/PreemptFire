import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from src.data_pipeline.accessibility.solver import dijkstra_kernel
from src.processing.grid_aligner import GridAligner


def fetch_copernicus_dem_10m(grid_info: dict) -> np.ndarray:
    """
    Streams Copernicus DEM (30m) via Planetary Computer STAC API
    and reprojects onto the 10m master UTM grid (2048x2048).
    """
    shape = grid_info["shape"]
    master_crs = grid_info["crs"]
    master_transform = grid_info["transform"]
    bbox_wgs84 = grid_info["bbox_wgs84"]

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = client.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox_wgs84,
        limit=1
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Copernicus DEM tiles found for target coordinates.")

    asset_url = items[0].assets["data"].href
    dem_10m = np.zeros(shape, dtype=np.float32)

    with rasterio.open(asset_url) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_10m,
            src_crs=src.crs,
            src_transform=src.transform,
            dst_crs=master_crs,
            dst_transform=master_transform,
            resampling=Resampling.bilinear,
            dst_nodata=0.0
        )

    return np.nan_to_num(dem_10m, nan=0.0)


def fetch_osm_infrastructure(grid_info: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Queries OSM roads, trails, and water barriers within the scene BBOX
    and rasterizes them directly onto the 10m master grid.
    """
    shape = grid_info["shape"]
    master_crs = grid_info["crs"]
    master_transform = grid_info["transform"]
    min_lon, min_lat, max_lon, max_lat = grid_info["bbox_wgs84"]

    # 1. Fetch Roads and Trails (Sources)
    road_tags = {"highway": True}
    if hasattr(ox, "features_from_bbox"):
        gdf_roads = ox.features_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat), tags=road_tags)
    else:
        gdf_roads = ox.geometries_from_bbox(max_lat, min_lat, max_lon, min_lon, tags=road_tags)

    sources = np.zeros(shape, dtype=np.uint8)
    if gdf_roads is not None and not gdf_roads.empty:
        gdf_roads = gdf_roads.to_crs(master_crs)
        shapes = [(geom, 1) for geom in gdf_roads.geometry if geom is not None and not geom.is_empty]
        if shapes:
            sources = rasterize(shapes=shapes, out_shape=shape, transform=master_transform, fill=0, dtype=np.uint8)

    # 2. Fetch Water Bodies (Impassable mask)
    water_tags = {"natural": "water", "waterway": ["river", "riverbank"]}
    try:
        if hasattr(ox, "features_from_bbox"):
            gdf_water = ox.features_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat), tags=water_tags)
        else:
            gdf_water = ox.geometries_from_bbox(max_lat, min_lat, max_lon, min_lon, tags=water_tags)
    except Exception:
        gdf_water = None

    passable_mask = np.ones(shape, dtype=np.uint8)
    if gdf_water is not None and not gdf_water.empty:
        gdf_water = gdf_water.to_crs(master_crs)
        water_shapes = [(geom, 0) for geom in gdf_water.geometry if geom is not None and not geom.is_empty]
        if water_shapes:
            passable_mask = rasterize(shapes=water_shapes, out_shape=shape, transform=master_transform, fill=1, dtype=np.uint8)

    return sources, passable_mask


def test_real_scene_accessibility(lat: float = 51.1784, lon: float = -115.5708):
    """
    Executes full 2048x2048 real-world accessibility test (Banff National Park, Alberta)
    and visualizes side-by-side comparison: Roads & Trails vs Optimal Walking Time.
    """
    aligner = GridAligner()
    grid_info = aligner.get_master_grid_info(lat=lat, lon=lon)
    resolution = 10.0

    print(f"1. Fetching Copernicus DEM 10m for ({lat}, {lon}) ...")
    elevation = fetch_copernicus_dem_10m(grid_info)

    print("2. Fetching OSM roads, trails, and water barriers ...")
    sources, passable_mask = fetch_osm_infrastructure(grid_info)

    print(f"3. Running Dijkstra-Tobler kernel on {elevation.shape} grid ...")
    optimal_time = dijkstra_kernel(
        elevation=elevation,
        sources=sources,
        passable_mask=passable_mask,
        resolution=resolution
    )

    # --- Side-by-side Visualization (2048 x 2048 Real Scene) ---
    print("4. Generating side-by-side visualization ...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Graph 1: Roads & Trails Mask
    im0 = axes[0].imshow(sources, cmap="gray_r", origin="upper")
    axes[0].set_title(f"Roads & Trails (OSM 10m)\nCoords: [{lat:.4f}, {lon:.4f}]", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="1 = Road / Trail, 0 = Forest")

    # Graph 2: Optimal Accessibility Time
    time_display = np.where(np.isinf(optimal_time), np.nan, optimal_time)
    cmap_time = plt.cm.plasma.copy()
    cmap_time.set_bad(color="black")  # Impassable water or infinite distance

    im1 = axes[1].imshow(time_display, cmap=cmap_time, origin="upper")
    axes[1].set_title("Optimal Travel Time (Dijkstra + Tobler)\nAccounting for Slopes & Barriers", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Walking Time [hours] (Black = Impassable / Infinite)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Real-world test in Banff National Park (Rocky Mountains, Alberta)
    test_real_scene_accessibility(lat=51.1784, lon=-115.5708)