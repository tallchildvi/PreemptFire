import matplotlib.pyplot as plt
import numpy as np
from src.processing.grid_aligner import GridAligner
from src.data_pipeline.population_fetcher import PopulationFetcher

aligner = GridAligner()
# grid_info = aligner.get_master_grid_info(lat=53.5461, lon=-113.4937)
grid_info = aligner.get_master_grid_info(lat=59.3862, lon=-108.8931627)


fetcher = PopulationFetcher()
pop_layer = fetcher.fetch_population(grid_info=grid_info)
pop_raw_layer = fetcher.fetch_raw_population(grid_info=grid_info)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(pop_layer, cmap="inferno", origin="upper")
axes[0].set_title("Settlement Density (log1p)")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(pop_raw_layer, cmap="inferno", origin="upper")
axes[1].set_title("Settlement Density Raw")
axes[1].axis("off")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# axes[1].hist(pop_layer.ravel(), bins=50, color="crimson", edgecolor="black")
# axes[1].set_title("Pixel Value Distribution")
# axes[1].set_xlabel("log1p(density)")
# axes[1].set_ylabel("Pixel Count")

plt.tight_layout()
plt.show()
