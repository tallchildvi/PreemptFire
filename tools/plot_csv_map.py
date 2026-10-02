import argparse
from pathlib import Path
import folium
import pandas as pd


def generate_points_map(csv_path: str, output_html: str):
    input_file = Path(csv_path)
    if not input_file.exists():
        print(f"Помилка: Файл {input_file} не знайдено.")
        return

    print(f"Читання даних з {input_file.name}...")
    df = pd.read_csv(input_file)

    if df.empty:
        print("Помилка: CSV файл порожній.")
        return

    # Підтримка різних варіантів назв колонок
    lat_col = "latitude" if "latitude" in df.columns else "lat"
    lon_col = "longitude" if "longitude" in df.columns else "lon"

    if lat_col not in df.columns or lon_col not in df.columns:
        print("Помилка: Не знайдено колонок координат (latitude/lat, longitude/lon).")
        return

    # Автоматичне центрування карти за медіаною всіх точок
    center_lat = df[lat_col].median()
    center_lon = df[lon_col].median()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Додавання супутникового шару
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
    ).add_to(m)

    # Шари для зручного увімкнення/вимкнення точок
    fg_fires = folium.FeatureGroup(name="Active Fires (1)", show=True)
    fg_negatives = folium.FeatureGroup(name="Background Negatives (0)", show=True)

    points_added = 0
    for _, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Пропускаємо пошкоджені рядки
        if pd.isna(lat) or pd.isna(lon):
            continue

        is_fire = int(row.get("is_fire", 1))
        acq_date = row.get("acq_date", row.get("target_date", "Unknown"))
        frp = row.get("frp", 0.0)
        
        if is_fire == 1:
            color = "#e11d48"  # Червоний
            fg = fg_fires
            label = "Fire"
        else:
            color = "#16a34a"  # Зелений
            fg = fg_negatives
            label = "Negative"

        popup_html = (
            f"<div style='min-width: 150px; font-family: sans-serif;'>"
            f"<b>Class:</b> {label} ({is_fire})<br>"
            f"<b>Date:</b> {acq_date}<br>"
            f"<b>FRP:</b> {frp}<br>"
            f"<b>Coords:</b> {lat:.5f}, {lon:.5f}"
            f"</div>"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{label}: {acq_date}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)
        
        points_added += 1

    fg_fires.add_to(m)
    fg_negatives.add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    
    print(f"Успішно нанесено {points_added} точок.")
    print(f"Інтерактивну карту збережено у: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive HTML map from points CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/master_points_iberia.csv",
        help="Path to the input CSV file containing lat/lon and is_fire columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/csv_points_map.html",
        help="Path to save the output interactive HTML map",
    )

    args = parser.parse_args()
    generate_points_map(csv_path=args.csv, output_html=args.output)