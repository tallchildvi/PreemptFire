import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import Window
import numpy as np

client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace
)

bbox = [-7.05, 41.65, -6.95, 41.75]

search_2021 = client.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2019-07-01/2019-07-31",
    query={"eo:cloud_cover": {"lt": 10}},
    limit=1
)
item_2021 = list(search_2021.items())[0]

search_2023 = client.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2023-07-01/2023-07-31",
    query={"eo:cloud_cover": {"lt": 10}},
    limit=1
)
item_2023 = list(search_2023.items())[0]

print("=" * 65)
print("1. ПЕРЕВІРКА МЕТАДАНИХ MICROSOFT STAC:")
print(f"2021 Date: {item_2021.datetime.date()} | Baseline: {item_2021.properties.get('s2:processing_baseline')}")
print(f"2023 Date: {item_2023.datetime.date()} | Baseline: {item_2023.properties.get('s2:processing_baseline')}")

b04_bands_2021 = item_2021.assets["B04"].extra_fields.get("raster:bands", [{}])[0]
b04_bands_2023 = item_2023.assets["B04"].extra_fields.get("raster:bands", [{}])[0]
print(f"2021 STAC registered offset: {b04_bands_2021.get('offset', 0)}")
print(f"2023 STAC registered offset: {b04_bands_2023.get('offset', 0)}")

print("\n" + "=" * 65)
print("2. ПЕРЕВІРКА РЕАЛЬНИХ ПІКСЕЛІВ ЧЕРЕЗ RASTERIO (СИРИЙ B02 / СИНІЙ КАНАЛ):")

with rasterio.open(item_2021.assets["B02"].href) as src:
    w = Window(src.width // 2, src.height // 2, 400, 400)
    data_2021 = src.read(1, window=w).astype(np.float32)

with rasterio.open(item_2023.assets["B02"].href) as src:
    w = Window(src.width // 2, src.height // 2, 400, 400)
    data_2023 = src.read(1, window=w).astype(np.float32)

v21 = data_2021[data_2021 > 0]
v23 = data_2023[data_2023 > 0]

print(f"2021 B02 (PB {item_2021.properties.get('s2:processing_baseline')}): min={np.min(v21):.1f}, 5th percentile={np.percentile(v21, 5):.1f}, median={np.median(v21):.1f}")
print(f"2023 B02 (PB {item_2023.properties.get('s2:processing_baseline')}): min={np.min(v23):.1f}, 5th percentile={np.percentile(v23, 5):.1f}, median={np.median(v23):.1f}")
print("=" * 65)