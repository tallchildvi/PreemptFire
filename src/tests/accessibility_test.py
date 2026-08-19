import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.data_pipeline.osm_dem_fetcher import SpatialFeatureFetcher

SCENES = {
    "ontario_boreal": (52.9607193, -91.3226589),
    "banff_alpine":   (51.1784,    -115.5708),
    "remote_shield":  (59.3862126, -108.8931627),
}


def _validate_layer(arr: np.ndarray, name: str) -> None:
    """Sanity-check a single output layer and print a one-line summary."""
    assert arr.ndim == 2,              f"{name}: expected 2D array, got shape {arr.shape}"
    assert arr.dtype == np.float32,    f"{name}: expected float32, got {arr.dtype}"
    assert not np.all(np.isnan(arr)),  f"{name}: all-NaN — fetch likely failed"

    finite = arr[np.isfinite(arr)]
    n_inf  = np.isinf(arr).sum()
    n_nan  = np.isnan(arr).sum()
    print(
        f"  {name:<22s} | shape {arr.shape} "
        f"| min {finite.min():7.3f}  max {finite.max():7.3f} "
        f"| inf {n_inf:6d}  nan {n_nan:6d}"
    )


def _validate_sources_mask(sources: np.ndarray) -> None:
    """Travel time == 0 marks pixels on roads/trails; verify it is non-empty."""
    n_sources = int((sources == 1).sum())
    coverage  = 100.0 * n_sources / sources.size
    print(f"  {'sources mask':<22s} | road/trail pixels: {n_sources:,d}  ({coverage:.2f}% of grid)")
    assert n_sources > 0, (
        "No road or trail pixels found — OSM query may have failed "
        "or the scene has zero infrastructure."
    )


def _validate_tobler_time(optimal_time: np.ndarray, cap_hours: float = 12.0) -> None:
    
    finite = optimal_time[np.isfinite(optimal_time)]
    assert (finite >= 0).all(),          "Negative travel times detected — Tobler logic error"
    assert (finite <= cap_hours + 1e-3).all(), (
        f"Travel time exceeds cap ({cap_hours} h) without being inf — clip missing"
    )
    reachable_pct = 100.0 * len(finite) / optimal_time.size
    print(f"  {'travel time':<22s} | reachable: {reachable_pct:.1f}%  "
          f"mean {finite.mean():.3f} h  median {np.median(finite):.3f} h")

def _plot_results(features: dict, lat: float, lon: float, scene_name: str, cap_hours: float = 12.0) -> None:
    travel_roads  = features["Travel_Time_Roads"]
    travel_trails = features["Travel_Time_Trails"]
    elevation     = features["Elevation"]
    slope         = features["Slope"]

    optimal_time  = np.minimum(travel_roads, travel_trails)
    sources_mask  = (optimal_time == 0.0).astype(np.uint8)

    # Impassable / water / capped pixels become NaN (rendered in black)
    time_display  = np.where(
        (optimal_time >= cap_hours) | np.isinf(optimal_time),
        np.nan,
        optimal_time,
    )

    cmap_time = plt.cm.plasma.copy()
    cmap_time.set_bad(color="black")

    # Dynamic vmax: scale colorbar to actual scene maximum reachable time
    finite_vals = time_display[np.isfinite(time_display)]
    max_time_scene = float(np.percentile(finite_vals, 99)) if len(finite_vals) > 0 else cap_hours
    time_clim = (0.0, max(max_time_scene, 0.5))

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f"SpatialFeatureFetcher — {scene_name}   ({lat:.5f}, {lon:.5f})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)

    panels = [
        (elevation,    "Elevation (m)",               "terrain",   None),
        (slope,        "Slope (°)",                   "YlOrRd",    None),
        (sources_mask, "Roads & Trails\n(OSM, 10 m)", "gray_r",    None),
        (time_display, "Optimal Travel Time\n(Dijkstra + Tobler, h)", cmap_time, time_clim),
    ]

    for idx, (arr, title, cmap, clim) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        kw = {"origin": "upper", "cmap": cmap}
        if clim is not None:
            kw["vmin"], kw["vmax"] = clim
        im = ax.imshow(arr, **kw)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
        if idx == 3:
            cbar.set_label(f"hours [0 - {time_clim[1]:.1f}h] (black = water/capped)", fontsize=8)
        elif idx == 2:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["no infra", "road/trail"])

    plt.tight_layout()
    plt.show()
    
def test_real_scene_accessibility(
    scene: str   = "ontario_boreal",
    lat:   float = None,
    lon:   float = None,
    cap_hours: float = 12.0,
    plot:  bool  = True,
) -> dict:
    
    if lat is None or lon is None:
        lat, lon = SCENES[scene]


    print(f"({lat:.6f}, {lon:.6f})")

    fetcher  = SpatialFeatureFetcher()
    features = fetcher.fetch_all_spatial_features(lat=lat, lon=lon)

    required = {
        "Elevation", "Slope", "Northness", "Eastness",
        "Travel_Time_Roads", "Travel_Time_Trails",
        "Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines",
    }
    missing = required - features.keys()
    assert not missing, f"Missing output keys: {missing}"

    print("\n[layer stats]")
    for key in sorted(required):
        _validate_layer(features[key], key)

    print("\n[accessibility checks]")
    travel_roads  = features["Travel_Time_Roads"]
    travel_trails = features["Travel_Time_Trails"]
    optimal_time  = np.minimum(travel_roads, travel_trails)

    sources_mask  = (optimal_time == 0.0).astype(np.uint8)
    _validate_sources_mask(sources_mask)
    _validate_tobler_time(optimal_time, cap_hours=cap_hours)

    if not np.all(travel_roads >= cap_hours) and not np.all(travel_trails >= cap_hours):
        identical = np.allclose(travel_roads, travel_trails, equal_nan=True)
        assert not identical, (
            "Travel_Time_Roads and Travel_Time_Trails are identical — "
            "likely both use the same source GeoDataFrame."
        )


    print("\nall assertions passed")

    if plot:
        _plot_results(features, lat, lon, scene, cap_hours=cap_hours)

    return features

if __name__ == "__main__":
    # test_real_scene_accessibility(scene="ontario_boreal")
    # test_real_scene_accessibility(scene="banff_alpine")
    test_real_scene_accessibility(scene="remote_shield")
