# PreemptFire: Multimodal GeoAI for Wildfire Susceptibility Forecasting

## Project Overview

PreemptFire is an end-to-end multimodal geospatial machine learning pipeline designed to assess spatiotemporal wildfire ignition susceptibility at $t+N$ days in advance.

The primary objective is pre-fire risk modeling rather than active fire perimeter tracking or post-fire damage assessment. The system ingests satellite imagery, polarimetric synthetic aperture radar (SAR), numerical weather reanalysis, physical fuel moisture indices, digital terrain metrics, and human infrastructure topology to predict where an ignition is most likely to occur and sustain itself.

```
                      +---------------------------------------+
                      |       Raw Geospatial Inputs           |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-----------------------------------+             +-----------------------------------+
|      2D Spatial Rasters           |             |    1D Meteorological Context      |
|    (Aligned UTM Grid @ 10 m/px)   |             |   (Open-Meteo & CFFDRS History)   |
+-----------------------------------+             +-----------------------------------+
| - Sentinel-2 Optical (B02-B12)    |             | - Hourly Surface Extremes         |
| - Sentinel-1 SAR (VV, VH)         |             | - Vapor Pressure Deficit (VPD)    |
| - Copernicus DEM (Slope, Aspect)  |             | - Continuous Haines Index         |
| - Distance to Roads / Trails (OSM)|             | - Sequential CFFDRS Codes         |
| - Soil Moisture & Population Count|             |   (FFMC, DMC, DC, ISI, BUI, FWI)  |
+-----------------------------------+             +-----------------------------------+
```

---

## Data Architecture and Modalities

All spatial rasters are orthorectified and bilinearly reprojected into a local UTM coordinate reference system on a unified master grid at a spatial resolution of $10\text{ m}$ per pixel.

### 1. Spatial Raster Channels (2D Tensors)

| Data Source         | Resolution | Channels / Variables             | Physical Role & Scientific Value                                       |
|:--------------------|:-----------|:---------------------------------|:-----------------------------------------------------------------------|
| Sentinel-2 L2A      | 10 m       | $B02\text{--}B12$, $SCL$         | Biomass density, vegetation moisture ($NDMI$, $NDVI$, $NBR$), desiccation|
| Sentinel-1 SAR RTC  | 10 m       | Polarizations $VV$, $VH$         | Cloud-penetrating surface roughness and duff layer moisture status     |
| Copernicus DEM      | 10 m       | Elevation, Slope, Aspect         | Topographic fire propagation vectors and solar radiation exposure      |
| OpenStreetMap (OSM) | 10 m       | Distance decay ($e^{-d/\sigma}$) | Anthropogenic ignition vectors (roads, power lines, campsites, trails) |
| ESA WorldCover      | 10 m       | Discrete Land Cover Classes      | Burnable fuel mask generation ($M_{\text{burnable}}$) and barrier mapping |
| ERA5-Land           | 10 m       | Soil Moisture Layer 1 ($0-7\text{ cm}$) | Background moisture baseline of shallow organic duff                   |
| WorldPop            | 10 m       | Residential Population Density   | Proxy for baseline human activity and accidental ignition hazard       |

### 2. Meteorological Context Vectors (1D Tabular)

Atmospheric variables are extracted via the Open-Meteo Archive API and ERA5 reanalysis. Continuous fuel codes and behavior metrics are computed via a historical 90-day sequential spin-up using the Canadian Forest Fire Danger Rating System (CFFDRS) equations (Van Wagner, 1987).

| Feature Category     | Source Variables                 | Extracted Metrics                | Physical Interpretation                                                |
|:---------------------|:---------------------------------|:---------------------------------|:-----------------------------------------------------------------------|
| Surface Extremes     | $T_{2m}$, $RH_{2m}$, Dew Point   | $T_{\max}$, $RH_{\min}$, $VPD$   | Atmospheric vapor deficit driving fuel drying rates                    |
| Wind Dynamics        | $U_{10m}$, $V_{10m}$, Gusts     | Mean Speed, Max Gust, Direction  | Convective oxygen supply and potential initial rate of spread          |
| Antecedent Drought   | Total Precipitation              | Consecutive Dry Days ($< 1.0\text{ mm}$) | Cumulative fuel drying duration prior to prediction timestamp  |
| Vertical Instability | Pressure levels (850, 700 hPa)   | Continuous Haines Index          | Plume-dominated fire growth potential in dry lower troposphere         |
| CFFDRS Fuel Moisture | Historical weather (90-day run)  | $FFMC$, $DMC$, $DC$              | Moisture levels across fine litter, duff layer, and deep organic soil  |
| CFFDRS Fire Behavior | Historical weather (90-day run)  | $ISI$, $BUI$, $FWI$              | Combined wind-fuel ignition sustainment, total fuel, and fire intensity|

### 3. Target Representations (Ground Truth)

Targets are derived from satellite thermal anomalies (NASA FIRMS VIIRS $375\text{ m}$ / MODIS) and national wildfire registries (e.g., Canadian National Fire Database). Positive ignitions undergo lookback filtering to ensure they represent initial outbreaks rather than ongoing burned perimeters.

Ignition centroids are splatted into multi-channel float32 Gaussian heatmaps and masked against non-burnable land covers ($M_{\text{burnable}}$):

| Target Channel | Kernel Radius ($\sigma$) | Theoretical Diameter ($6\sigma$) | Spatial Purpose                                         |
|:---------------|:-------------------------|:---------------------------------|:--------------------------------------------------------|
| Channel 0      | $\sigma = 250\text{ m}$  | $1.5\text{ km}$                  | Pinpoint ignition footprint                             |
| Channel 1      | $\sigma = 1000\text{ m}$ | $6.0\text{ km}$                  | Sub-regional spatial hazard envelope                    |
| Channel 2      | $\sigma = 3000\text{ m}$ | $18.0\text{ km}$                 | Landscape-scale susceptibility corridor                 |
| Channel 3      | $\sigma = 4000\text{ m}$ | $24.0\text{ km}$                 | Macro-regional risk field                               |

*Note: For confirmed non-fire samples, all target channels are populated with exact zero tensors.*

---

## Leakage Prevention and Spatial Independence

To ensure rigorous evaluation and prevent spatial autocorrelation leakage, data partitioning adheres to strict protocols:

1. **Blocked Spatial Splitting:** Complete scene patches ($2048 \times 2048\text{ px}$) are partitioned geographically. Spatial independence is validated using Global Moran's $I$ ($I \approx 0$).
2. **Temporal Splitting:** Training partitions strictly precede validation and testing partitions in historical time, eliminating future information leakage into fuel moisture models.
3. **No Target-Day Weather in Predictor Inputs:** When evaluating predictions for horizon $t+N$, meteorological fuel states are computed strictly from information available at or prior to observation timestamp $t_0$.

---

## Project Structure

```text
preemptfire/
|-- data/
|   |-- raw/                    # Raw FIRMS CSVs and administrative GeoJSONs
|   |-- interim/                # Pre-filtered points and spatial check artifacts
|   `-- processed_h5/           # Final extracted HDF5 patch archives
|-- docs/                       # Technical reports and analytical documentation
|-- notebooks/                  # Exploratory spatial data analysis (EDA)
|-- src/
|   |-- config.py               # Master grid dimensions and projection parameters
|   |-- data_pipeline/          # Fetchers for FIRMS, STAC (S1/S2), weather, and OSM
|   |-- processing/             # Spatial reprojection, UTM alignment, and CFFDRS engine
|   |-- target_pipeline/        # Multi-scale Gaussian splatting and land cover masking
|   `-- utils/                  # Spatial thinning, Haversine metrics, and spatial statistics
|-- requirements.txt            # Python environment specifications
`-- README.md                   # Primary project documentation
```

---

## Technology Stack

- **Geospatial Processing:** GDAL, Rasterio, GeoPandas, PySTAC Client, PyOsmium, OSMnx
- **Numerical and Physics Modeling:** NumPy, SciPy, Numba, Van Wagner CFFDRS Engine
- **Machine Learning Frameworks:** PyTorch, TorchGeo, CatBoost
- **Data Persistence:** HDF5 (h5py) with LZF compression and memory-efficient chunking