import math
import matplotlib.pyplot as plt
import numpy as np

from src.data_pipeline.single_scene_collector import SingleSceneCollector
from src.data_pipeline.patch_extractor import PatchExtractor


def print_scene_summary(sample: dict):
    meta = sample["metadata"]
    c1d = sample["context_1d"]

    print("\n" + "=" * 70)
    print(" SCENE METADATA & TABULAR 1D CONTEXT (Shared across all patches)")
    print("=" * 70)
    print(f"  Target Date       : {meta['target_date']}")
    print(f"  T0 Date (Tprev)   : {meta['t0_date']} ({meta['tprev_date']})")
    print(f"  Center Coordinates: [{meta['lat']:.4f}, {meta['lon']:.4f}] ({meta['crs']})")
    print(f"  Ignition Status   : {'FIRE (1)' if meta['is_fire'] else 'NO FIRE (0)'}")
    print(f"  Collection Time   : {meta.get('elapsed_seconds', 0.0):.2f} s")
    print("-" * 70)
    print("  1D TABULAR CONTEXT PARAMETERS:")
    print("-" * 70)
    for k, v in sorted(c1d.items()):
        print(f"    {k:<28}: {v:8.3f}")
    print("=" * 70 + "\n")


def plot_patch_fullscreen(patch: dict, patch_num: int, total_valid_patches: int = None):
    channel_names = patch["channel_names_2d"]
    x_2d = patch["X_2d"]
    loss_mask = patch["loss_mask"]
    target = patch["Y"]
    meta = patch["metadata"]

    plot_items = []

    # 1. Feature 2D layers
    for idx, name in enumerate(channel_names):
        arr = x_2d[idx]
        if "MASK" in name:
            cmap = "gray"
        elif "SAR" in name:
            cmap = "plasma"
        elif any(k in name for k in ["NDVI", "EVI", "NDRE"]):
            cmap = "RdYlGn"
        elif any(k in name for k in ["NDMI", "Soil_Moisture", "Water"]):
            cmap = "Blues"
        elif any(k in name for k in ["Slope", "Elevation", "Travel_Time", "Dist_"]):
            cmap = "terrain"
        else:
            cmap = "viridis"
        plot_items.append((f"{name}", arr, cmap))

    # 2. Loss mask
    plot_items.append(("Loss Mask (MASK_INVALID)", loss_mask, "binary_r"))

    # 3. 4-Channel Target Gaussians
    target_names = ["Target (σ=250m)", "Target (σ=1000m)", "Target (σ=3000m)", "Target (σ=4000m)"]
    target_cmaps = ["inferno", "magma", "plasma", "viridis"]
    for i in range(target.shape[0]):
        plot_items.append((target_names[i], target[i], target_cmaps[i]))

    # Grid calculations
    total_plots = len(plot_items)
    cols = 8
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 2.6 * rows))
    axes = axes.flatten()

    for idx, (title, img, cmap) in enumerate(plot_items):
        ax = axes[idx]
        im = ax.imshow(img, cmap=cmap, origin="upper")
        ax.set_title(f"{title}\n[{np.nanmin(img):.2f}, {np.nanmax(img):.2f}]", fontsize=7.5, pad=2)
        ax.axis("off")

    # Hide extra unused subplots
    for idx in range(total_plots, len(axes)):
        axes[idx].axis("off")

    title_suffix = f" (Patch #{patch_num})" if total_valid_patches is None else f" ({patch_num}/{total_valid_patches})"
    fig.suptitle(
        f"Patch ID: {patch['patch_id']}{title_suffix} | "
        f"Offset: [R:{meta['row_offset']}, C:{meta['col_offset']}] | "
        f"Invalid Ratio: {meta['invalid_ratio']*100:.1f}% | "
        f"Max Risk Target: {meta['target_max_local']:.4f}\n"
        f"(Close this window to view the next patch...)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    # Maximize window across operating systems
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized()
    except Exception:
        try:
            mng.frame.Maximize(True)
        except Exception:
            pass

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.15)
    plt.show()
    plt.close(fig)


def run_pipeline_patch_test(
    lat: float = 56.7264,
    lon: float = -111.3803,
    target_date: str = "2023-06-15",
    is_fire: int = 1,
    patch_size: int = 256,
    stride: int = 256,
    max_invalid_ratio: float = 0.20,
):
    collector = SingleSceneCollector()
    extractor = PatchExtractor(
        patch_size=patch_size,
        stride=stride,
        max_invalid_ratio=max_invalid_ratio,
    )

    sample = collector.collect_sample(
        lat=lat,
        lon=lon,
        target_date=target_date,
        is_fire=is_fire,
    )

    if not sample:
        print("[Error] Failed to collect master scene.")
        return

    # Print 1D metadata/weather/CFFDRS once
    print_scene_summary(sample)

    scene_id = f"SCENE_{lat:.2f}_{lon:.2f}_{target_date.replace('-', '')}"
    patch_generator = extractor.extract_patches(sample, scene_id=scene_id)

    print(f"Starting sequential patch visualization (Patch Size: {patch_size}x{patch_size})...\n")
    
    patch_idx = 0
    for patch in patch_generator:
        patch_idx += 1
        print(
            f"  --> Displaying {patch['patch_id']} | "
            f"Shape: {patch['X_2d'].shape} | "
            f"Invalid: {patch['metadata']['invalid_ratio']*100:.1f}%"
        )
        plot_patch_fullscreen(patch, patch_num=patch_idx)

    print(f"\nAll valid patches ({patch_idx} total) have been reviewed.")


if __name__ == "__main__":
    run_pipeline_patch_test(
        lat=56.7264,
        lon=-111.3803,
        target_date="2023-06-15",
        is_fire=1,
        patch_size=256,
        stride=256,
        max_invalid_ratio=0.20,
    )