# PreemptFire: Multimodal GeoAI for Wildfire Susceptibility Forecasting

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces%20Demo-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/tallchildvi/PreemptFire)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-%233776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

**Machine Learning and Modeling:**  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-%23FFCC00.svg?style=flat&logoColor=black)](https://catboost.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%238CAAE6.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)

**Earth Observation and Data Pipelines:**  
[![Copernicus Sentinel & DEM](https://img.shields.io/badge/Copernicus-Sentinel%201%2F2%20%7C%20DEM-%23004F9F.svg?style=flat)](https://dataspace.copernicus.eu/)
[![NASA FIRMS](https://img.shields.io/badge/NASA-FIRMS%20VIIRS-%23E03C31.svg?style=flat&logo=nasa&logoColor=white)](https://firms.modaps.eosdis.nasa.gov/)
[![Open-Meteo ERA5](https://img.shields.io/badge/Open--Meteo-ERA5%20Reanalysis-%23F39C12.svg?style=flat)](https://open-meteo.com/)
[![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-Infrastructure-%237EBC6F.svg?style=flat&logo=openstreetmap&logoColor=white)](https://www.openstreetmap.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-WorldPop-%234285F4.svg?style=flat&logo=googleearth&logoColor=white)](https://earthengine.google.com/)
[![EFFIS GWIS](https://img.shields.io/badge/EFFIS-GWIS%20CFFDRS-%23990000.svg?style=flat)](https://effis.jrc.ec.europa.eu/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-%23FD7E14.svg?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)

---

## Interactive Demonstration

The full multi-model forecasting pipeline and benchmarking suite is deployed on Hugging Face Spaces:
- Application URL: [https://huggingface.co/spaces/tallchildvi/PreemptFire](https://huggingface.co/spaces/tallchildvi/PreemptFire)
- Direct Mirror: [https://tallchildvi-preemptfire.hf.space](https://tallchildvi-preemptfire.hf.space)

The space provides synoptic risk mapping across 3,146 regional nodes over the Iberian Peninsula, side-by-side model comparison, 33-channel multimodal raster inspection, and independent post-hoc verification against live NASA FIRMS VIIRS ground-truth satellite detections.

---

## Project Overview

PreemptFire is an end-to-end multimodal geospatial machine learning framework designed to forecast wildfire ignition susceptibility and expected fire counts across the Iberian Peninsula (Spain and Portugal) at lead times of $t+0$ to $t+10$ days.

Unlike systems focused on active fire perimeter tracking or post-fire burn scar assessment, PreemptFire addresses pre-fire risk modeling. It integrates optical surface reflectance, polarimetric synthetic aperture radar (SAR), digital elevation metrics, friction-weighted road networks, numerical atmospheric reanalysis, demographic pressure, and multi-year fire history to predict where ignitions are physically and empirically most likely to occur.

```
                      +-------------------------------------------------+
                      |            Raw Geospatial Data Streams          |
                      +------------------------+------------------------+
                                               |
         +-------------------------------------+-------------------------------------+
         |                                                                           |
         v                                                                           v
+---------------------------------------+                   +---------------------------------------+
|        2D Spatial Scene Rasters       |                   |      1D Meteorological & History      |
|    (128 x 128 px @ 100 m/px, 33 ch)   |                   |            Context Vectors            |
+---------------------------------------+                   +---------------------------------------+
| - Sentinel-2 Optical (B02-B12, SCL)   |                   | - Open-Meteo ERA5 Reanalysis          |
| - Sentinel-1 SAR RTC (VV, VH)         |                   |   (T_min/max, RH, VPD, Wind, Precip)  |
| - Vegetation Indices (NDVI, NDMI, ...) |                   | - Antecedent Drought & Drying Memory  |
| - Copernicus DEM (Slope, Aspect)      |                   | - WorldPop Demographics (10 km, 50 km)|
| - OSM Road/Trail Travel Time Surfaces |                   | - NASA FIRMS 30-Day Spatiotemporal    |
| - Distances to Rail, Camps, Powerlines|                   |   History (5, 10, 25, 50 km radii)    |
| - ERA5-Land Volumetric Soil Moisture  |                   | - 9-Year Historical Day-of-Year       |
+---------------------------------------+                   |   Fire Climatology (2017-2025)        |
                                                            +---------------------------------------+
                                               |
                                               v
                      +-------------------------------------------------+
                      |           Model Evaluation & Inference          |
                      +-------------------------------------------------+
                      | - tab_clim_full (CatBoost + Climatology, 69 ft) |
                      | - tab_ssl_full (CatBoost + SimCLR ResNet, 101 ft)|
                      | - tab_full (CatBoost + Handcrafted 2D, 262 ft)  |
                      | - nn_ts (PyTorch GRU Time Series + Static)      |
                      | - Physical Baseline: EFFIS CFFDRS FWI (ecmwf)   |
                      | - Ensembles: stack_no_img, stack_all            |
                      +-------------------------------------------------+
```

---

## Data Architecture and Modalities

The target domain encompasses mainland Spain, Portugal, and the Balearic Islands bounded by coordinates $[-9.5^\circ\text{W}, 35.9^\circ\text{N}, 3.4^\circ\text{E}, 43.9^\circ\text{N}]$. The master grid evaluates 3,146 regional spatial nodes at a native resolution of $100\text{ m}$ per pixel.

### 1. Spatial Raster Channels (2D Modality, 33 Channels)

Spatial rasters are constructed as $128 \times 128$ pixel scenes ($12.8 \times 12.8\text{ km}$ spatial footprint) centered over each target node:

| Channel Group | Variables / Bands | Data Source | Physical and Analytical Role |
| :--- | :--- | :--- | :--- |
| **Optical Surface Reflectance** | $B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, SCL$ | Sentinel-2 L2A (Copernicus) | Canopy structure, chlorophyll absorption, and cloud/water screening |
| **Spectral Vegetation Indices** | $NDVI, NDMI, NBR, NBR2, EVI, NDRE, MSI, NMDI$ | Computed from Sentinel-2 | Live fuel moisture content, foliar desiccation, and biomass density |
| **SAR Backscatter** | $VV, VH$, Cross-Ratio ($VH/VV$), Radar Vegetation Index ($RVI$) | Sentinel-1 SAR RTC (Copernicus) | Cloud-penetrating surface roughness, duff structure, and moisture |
| **Topography** | Elevation, Slope, Northness, Eastness | Copernicus DEM GLO-30 | Orographic fire propagation vectors, solar insolation, and wind channeling |
| **Anthropogenic Infrastructure** | Travel Time to Roads, Travel Time to Trails | OpenStreetMap (OSM) | Friction surfaces capturing human accessibility and ignition access |
| **Linear Hazard Vectors** | Distance to Railways, Distance to Campsites, Distance to Powerlines | OpenStreetMap (OSM) | Euclidean proximity to known human ignition risk corridors |
| **Subsurface Moisture** | Volumetric Soil Water Layer 1 ($0\text{--}7\text{ cm}$) | ERA5-Land (ECMWF) | Deep organic duff moisture baseline mitigating false surface drying |

### 2. Meteorological and Spatiotemporal Context (1D Modality)

Atmospheric states are retrieved dynamically via the Open-Meteo ERA5 / ERA5T reanalysis API, complemented by demographic layers and active fire records:

- **Surface Weather Extremes (35 base features):**
  - Daily extrema: $T_{\min}, T_{\max}, T_{\text{mean}}$, relative humidity $RH_{\min}, RH_{\text{mean}}$, dew point temperature.
  - Atmospheric dryness: Vapor Pressure Deficit ($VPD_{\max}, VPD_{\text{mean}}$) and mean surface atmospheric pressure.
  - Wind dynamics: 10m mean wind speed, maximum wind gusts, and circular wind direction components ($\cos(\theta), \cos(2\theta)$).
  - Target-day precipitation totals.
- **Drought and Desiccation Memory:**
  - Antecedent drying: cumulative precipitation over 7-day and 30-day lookback windows.
  - Thermal accumulation: mean and maximum temperatures over 7-day and 30-day windows.
  - Consecutive dry days ($P < 1.0\text{ mm}$) and temporal gap between Sentinel-2 observation pass ($t_0$) and forecast date.
- **Demographic Density:**
  - WorldPop residential population counts within $10\text{ km}$ and $50\text{ km}$ radial envelopes ($\log(1+p)$ transformed).
- **Spatiotemporal Fire History (NASA FIRMS VIIRS 375m & MODIS):**
  - Historical active fire count matrices evaluated across 4 concentric radii ($R \in \{5, 10, 25, 50\}\text{ km}$) and 5 antecedent windows ($W \in \{1, 3, 7, 14, 30\}\text{ days}$).
- **Historical Climatology Baseline:**
  - 9-year historical day-of-year fire frequency extracted from 2017 to 2025 FIRMS records across a $\pm 15$-day seasonal window.
- **Physical Fire Weather Index Reference:**
  - Dynamic Canadian Forest Fire Danger Rating System (CFFDRS) FWI rasters ingested via the Copernicus EFFIS / GWIS WMS service (`ecmwf.fwi`).

---

## Machine Learning Architecture

The repository implements multiple machine learning paradigms, ranging from gradient-boosted decision trees to self-supervised deep visual encoders and recurrent architectures:

| Model Identifier | Architecture | Input Feature Set | Description |
| :--- | :--- | :--- | :--- |
| **`tab_clim_full`** | CatBoost GBDT | 69 features | Weather, 30-day FIRMS history, DoY climatology, and demographic vectors. **Primary benchmark model.** |
| **`tab_ssl_full`** | ResNet SimCLR + CatBoost | 101 features | CatBoost trained on 1D features augmented with 32-dimensional dense visual representations from a self-supervised SimCLR ResNet encoder trained on 33-channel satellite scenes. |
| **`tab_full`** | CatBoost GBDT | 262 features | Combines 1D weather features with handcrafted spatial and spectral summary statistics (mean, std, 10th/90th percentiles, central spatial crops) across all 33 raster channels. |
| **`tab_hist`** | CatBoost GBDT | 60 features | Meteorological features combined strictly with 30-day FIRMS history (excluding multi-year climatology). |
| **`tab_1d`** | CatBoost GBDT | 39 features | Pure atmospheric and demographic model without spatial fire history. |
| **`nn_ts`** | PyTorch GRU | Temporal sequence | Recurrent neural network processing 30-day temporal sequences of weather and FIRMS fire counts, concatenated with static demographic and terrain features. |
| **`coords`** | Spatial k-NN | Latitude, Longitude | Non-parametric spatial nearest-neighbor baseline. |
| **`persist`** | Persistence | Historical detections | Evaluates fire persistence based on active detections from the preceding 7 days. |
| **`clim`** | Climatology | Day-of-year | Pure historical fire climatology baseline based on the 2017–2025 FIRMS archive. |
| **`fwi`** | EFFIS / GWIS | ECMWF CFFDRS FWI | Standard Canadian Forest Fire Weather Index physical baseline queried from the Copernicus WMS service. |
| **`stack_no_img`** | Stacking Ensemble | Tabular + Recurrent | Blended ensemble combining non-image models (`tab_clim_full`, `tab_hist`, `nn_ts`, `clim`). |
| **`stack_all`** | Stacking Ensemble | Multimodal | Full multimodal ensemble combining visual, recurrent, and tabular models. |

---

## Empirical Findings: The Physical FWI Paradox in Mediterranean Ecosystems

A core scientific finding demonstrated in this repository is the limitation of relying solely on physical fire weather indices (such as the Canadian Fire Weather Index, FWI) in Mediterranean environments without accounting for vegetation fuel availability and human factors.

### Quantitative Comparison (Evaluation Target: August 14, 2026, $R = 50\text{ km}$)

| Evaluation Metric | `tab_clim_full` (Machine Learning) | `fwi` (Physical EFFIS Baseline) | Scientific Implication |
| :--- | :---: | :---: | :--- |
| **ROC-AUC (Target Day)** | **0.742** | **0.422** | FWI performs worse than random chance ($< 0.50$); ML discriminates accurately |
| **ROC-AUC (7-day window)** | **0.712** | **0.476** | FWI fails to anticipate multi-day fire clustering |
| **Fire Hit Rate: Low Tier** | **8.8%** (149 / 1,690) | **21.0%** (397 / 1,887) | FWI assigns the majority of actual ignitions to its lowest tier |
| **Fire Hit Rate: Medium Tier** | **22.9%** (226 / 986) | **19.8%** (187 / 944) | — |
| **Fire Hit Rate: High Tier** | **48.3%** (227 / 470) | **5.7%** (18 / 315) | **Complete physical inversion**: FWI High zone contained minimal ignitions |
| **Inter-Model Correlation** | — | — | **$r = 0.080$** (near-zero spatial correlation) |

### Physical Interpretation

1. **Atmospheric Potential vs. Fuel Availability:**  
   The Canadian FWI assumes a continuous, standardized pine fuel bed. During Mediterranean heatwaves, atmospheric temperatures in central and southeastern Spain (Castilla-La Mancha, Extremadura plains, Murcia) exceed $38^\circ\text{C}$ with relative humidity below $15\%$, producing extreme FWI values ($>50\dots 70$). However, these areas consist of arid steppes, non-vegetated plateaus, or post-harvest agricultural stubble where burnable fuel is scarce.
2. **Where Wildfires Actually Occur:**  
   Actual active ignitions (validated via satellite VIIRS thermal anomalies) occur predominantly in the forested, fuel-rich corridors of northwestern Iberia (Galicia, northern Portugal, and Cantabrian ranges). Although these oceanic-influenced areas exhibit lower air temperatures and higher humidity (yielding Low-to-Moderate FWI ratings), they maintain dense, highly flammable fuel loads (eucalyptus, maritime pine, and ericaceous shrubs) coupled with high human interface pressure.
3. **Machine Learning Solution:**  
   The multimodal ML model (`tab_clim_full`) resolves this discrepancy by conditioning atmospheric drought on empirical fuel history, day-of-year climatological probability, and infrastructure proximity, producing monotonic risk stratification:
   $$\text{Low (8.8\%)} \longrightarrow \text{Medium (22.9\%)} \longrightarrow \text{High (48.3\%)}$$

---

## Repository Structure

```text
PreemptFire/
|-- hf_space/                           # Hugging Face Spaces web application
|   |-- app.py                          # 4-tab Gradio interactive dashboard
|   |-- config_and_features.py          # Feature calculators, normalizers, and config
|   |-- data_collectors.py              # Open-Meteo ERA5 fetcher, NASA FIRMS client, 2D collector
|   |-- fwi_calculator.py               # Copernicus EFFIS GWIS WMS integration (ecmwf.fwi)
|   |-- model_registry.py               # Model catalog, weight loaders, and inference engine
|   |-- post_hoc_evaluation.py          # Ground-truth validation against VIIRS detections
|   |-- mini_dataset_store.py           # Sharded persistence and cache management
|   |-- metadata.csv                    # 3,146 Iberian evaluation coordinates
|   `-- requirements.txt                # Deployment dependencies for Gradio Space
|-- src/
|   |-- config.py                       # Global grid parameters (128x128 px @ 100m)
|   |-- data_pipeline/                  # Offline data extraction engines
|   |   |-- single_scene_collector.py   # Multi-sensor collector (Sentinel 1/2, DEM, OSM, Weather)
|   |   |-- sentinel_fetcher.py         # STAC client for Sentinel-2 L2A & Sentinel-1 SAR
|   |   |-- osm_dem_fetcher.py          # Topography and OSM travel-time friction surfaces
|   |   |-- weather_fetcher.py          # Open-Meteo ERA5 historical reanalysis parser
|   |   |-- population_fetcher.py       # Demographic density extractor (WorldPop)
|   |   `-- generate_dataset.py         # Multi-worker shard generator with SQLite tracking
|   |-- modeling/                       # Model training and offline evaluation
|   |   |-- train_catboost.py           # CatBoost baseline training scripts
|   |   `-- dataset.py                  # PyTorch Dataset for multimodal shards
|   `-- processing/
|       `-- grid_aligner.py             # Spatial alignment and coordinate reprojection
|-- requirements.txt                    # Full Python environment specification
|-- LICENSE                             # MIT License
`-- README.md                           # Master technical documentation
```

---

## Installation and Local Execution

### 1. Environment Setup

Clone the repository and install dependencies in a Python 3.10 or 3.11 environment:

```bash
git clone https://github.com/tallchildvi/PreemptFire.git
cd PreemptFire
pip install -r requirements.txt
```

### 2. Running the Interactive Application Locally

To run the interactive Gradio dashboard locally:

```bash
cd hf_space
python app.py
```

The web interface will launch at `http://127.0.0.1:7860`.

### 3. Environment Variables (Optional)

To enable live NASA FIRMS API queries without falling back to pre-cached historical archives, configure an API key:

```bash
export FIRMS_MAP_KEY="your_nasa_firms_key"
```

For Google Earth Engine demographic extractions (offline pipeline):

```bash
export GEE_PROJECT="your_gee_project_id"
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.