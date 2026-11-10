---
title: PreemptFire Iberia Wildfire Risk and Model Comparison
emoji: globe
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# PreemptFire: Iberia Wildfire Risk and Model Comparison

Gradio application for Hugging Face Spaces. Runs data collection following the approach_b branch and evaluates models trained in preemptfire-v5_(2).ipynb.

## Repository structure

- `app.py`: Gradio web interface (Iberia risk map, model comparison, live 2D scene collection, mini-dataset browser).
- `make_map.py`: CLI script to compute risk maps for the Iberia grid (3,146 locations).
- `live.py`: CLI script for single-coordinate evaluation and 22-channel 256x256 patch collection.
- `config_and_features.py`: feature calculation routines for TAB_1D (39 features), HIST_TAB (21 features), CLIM_FEATS (9 features), SEQ and STATIC_1D for nn_ts, FT (202 features), and 33-channel tensor construction for SimCLR.
- `data_collectors.py`: batched Open-Meteo ERA5 weather fetcher, NASA FIRMS VIIRS reader, and Scene2DCollector (Sentinel-2, Sentinel-1, DEM, OSM road travel time, soil moisture).
- `model_registry.py`: model loader and inference engine for tab_ssl_full, tab_full, tab_clim_full, tab_hist, tab_1d, nn_ts, coords, persist, clim, and ensembles.
- `mini_dataset_store.py`: persistence manager for collected .npz scenes, daily feature matrices, model predictions, and zip export.
- `prepare_hf_bundle.py`: pre-deployment helper to verify and stage model files and caches.

## Deployment to Hugging Face Spaces

1. Stage model weights and supporting data:
   ```bash
   python prepare_hf_bundle.py \
       --output-dir . \
       --models-src /path/to/preemptfire_full_output \
       --firms-gz /path/to/firms_points_2017_2025.csv.gz \
       --scenes-dir /path/to/scenes_npz
   ```
   If `--scenes-dir` is provided, the script extracts 10 km and 50 km population values into `population_cache.csv`, removing the need for Google Earth Engine credentials.

2. Create a Space on Hugging Face (SDK: Gradio, Hardware: CPU basic).
3. Push files using Git LFS:
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/<USER>/<SPACE_NAME>
   cp -r hf_space/* <SPACE_NAME>/
   cd <SPACE_NAME>
   git lfs track "*.cbm" "*.pt" "*.pkl" "*.gz" "*.npz"
   git add .gitattributes
   git add .
   git commit -m "Deploy PreemptFire app"
   git push
   ```

4. Configure environment secrets in Space settings if needed:
   - `FIRMS_MAP_KEY`: NASA FIRMS API key (only required for dates after 2025).
   - `GEE_PROJECT`, `GEE_SERVICE_ACCOUNT`, `GEE_KEY_FILE`: optional fallback if not using `population_cache.csv`.
