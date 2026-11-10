---
title: PreemptFire Iberia Wildfire Risk & Model Comparison
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# 🔥 PreemptFire: Карта Пожежної Небезпеки Іберії та Порівняння Моделей (1D + 2D Супутникові Знімки)

Повний набір скриптів та інтерактивний веб-додаток **Gradio** для **Hugging Face Spaces**, побудований у точній відповідності до гілки [`approach_b`](https://github.com/tallchildvi/PreemptFire/tree/approach_b) та ноутбука експериментів `preemptfire-v5_(2).ipynb`.

---

## 📂 Структура папки `hf_space/`

- `app.py` — головний веб-інтерфейс **Gradio** для Hugging Face Spaces (4 вкладки: карта Іберії + візуальне порівняння моделей, точковий live-збір 22-канальних 2D супутникових сцен `Sentinel-2 + S1 + DEM + OSM + ERA5`, перегляд та `.zip`-експорт міні-датасету, діагностика файлів).
- `make_map.py` — CLI-скрипт побудови карти ризику та візуального порівняння моделей по всій сітці Іберії (3 146 локацій з `metadata.csv`).
- `live.py` — CLI-скрипт точкового прогнозу для довільних `(lat, lon, date)` з підтримкою прапорця `--collect-2d` (збирає 22-канальний куб `256×256` і запускає включно з `tab_ssl_full` та `tab_full`).
- `config_and_features.py` — точні формули всіх ознак із ноутбука `preemptfire-v5_(2).ipynb`:
  - `TAB_1D` (39 метеорологічних, календарних та демографічних ознак),
  - `HIST_TAB` (21 ознака історії пожеж за 1, 3, 7, 14, 30 днів у радіусах 5, 10, 25, 50 км),
  - `CLIM_FEATS` (9 ознак багаторічної DOY-кліматології),
  - `SEQ` (30×5 часовий ряд) + `STATIC_1D` (47 ознак для PyTorch GRU `nn_ts`),
  - `compute_2d_tab_features` (202 просторові/спектральні статистики `FT` для `tab_full`),
  - `build_33ch_array` (33-канальний тензор `128×128` для `SimCLR` енкодера в `tab_ssl_full`).
- `data_collectors.py` — оптимізовані збирачі даних для Hugging Face Spaces:
  - Пакетний (multi-coordinate) запит **Open-Meteo ERA5** із дисковим кешем (замість 280 окремих запитів виконує лише ~7 пакетних запитів, уникаючи помилки `HTTP 429 Too Many Requests` та помилки `HTTP 400 era5_seamless`),
  - Завантажувач **NASA FIRMS VIIRS** (використовує `firms_points_2017_2025.csv.gz` для дат до кінця 2025 року + кешований API для нових дат),
  - Автономний збирач 22-канальних 2D-сцен **`Scene2DCollector`** (`Sentinel-2 L2A`, `Sentinel-1 RTC`, `Copernicus DEM GLO-30`, `OpenStreetMap Overpass + Dijkstra Travel Time`, `Soil Moisture`), інтегрований із `src/data_pipeline/` гілки `approach_b`.
- `model_registry.py` — єдиний реєстр та інференс усіх **11 моделей** (`tab_ssl_full`, `tab_full`, `tab_clim_full`, `tab_hist`, `tab_1d`, `nn_ts`, `coords`, `persist`, `clim`, `stack_no_img`, `stack_all`).
- `mini_dataset_store.py` — менеджер автоматичного збереження зібраних `.npz` сцен (`mini_dataset/scenes/`), `mini_dataset/metadata.csv`, денних матриць ознак (`mini_dataset/features/`) та прогнозів (`mini_dataset/predictions/`) із можливістю завантаження одним `.zip` архівом.
- `prepare_hf_bundle.py` — допоміжний скрипт для підготовки та перевірки всіх файлів перед деплоєм.

---

## 🚀 Покрокова інструкція розгортання на Hugging Face Spaces

### Крок 1. Підготуйте файли моделей локально
Скопіюйте всі файли з вашої папки `preemptfire_full_output/models/` у підпапку `hf_space/models/`, а файли `oof_predictions.csv`, `count_metrics.csv`, `summary.json` та `firms_points_2017_2025.csv.gz` — у корінь `hf_space/` (файли `metadata.csv` та `firms_covered_days.csv` вже додані у `hf_space/`).

Або просто запустіть скрипт автоматичної підготовки:
```bash
cd hf_space
python prepare_hf_bundle.py \
    --output-dir . \
    --models-src /шлях/до/preemptfire_full_output \
    --firms-gz /шлях/до/firms_points_2017_2025.csv.gz \
    --scenes-dir /шлях/до/локальних/scenes_npz
```
> **Підказка щодо населення (`population_cache.csv`):** якщо ви вкажете `--scenes-dir` (де лежать ваші тренувальні `.npz` файли), скрипт миттєво витягне `population_10km_log1p` та `population_50km_log1p` для всіх 3 146 точок у файл `population_cache.csv`. Тоді на Hugging Face Spaces **взагалі не знадобиться Google Earth Engine**!

### Крок 2. Створіть новий Space на Hugging Face
1. Перейдіть на [huggingface.co/new-space](https://huggingface.co/new-space).
2. Вкажіть назву (наприклад, `PreemptFire-Iberia`), оберіть **Space SDK: `Gradio`**, Hardware: **`CPU basic (Free)`**.
3. Оскільки частина файлів (`.cbm`, `.pt`, `firms_points_2017_2025.csv.gz`) більші за 10 МБ, увімкніть **Git LFS** та завантажте вміст папки `hf_space/`:
```bash
git lfs install
git clone https://huggingface.co/spaces/<ВАШ_ЛОГІН>/<НАЗВА_SPACE>
cp -r hf_space/* <НАЗВА_SPACE>/
cd <НАЗВА_SPACE>
git lfs track "*.cbm" "*.pt" "*.pkl" "*.gz" "*.npz"
git add .gitattributes
git add .
git commit -m "Deploy PreemptFire multi-model app (1D + 2D satellite models)"
git push
```

### Крок 3. Додайте секрети у налаштуваннях Space
У вашому Space відкрийте **Settings → Variables and secrets → New secret**:
- `FIRMS_MAP_KEY` — ваш ключ NASA FIRMS API (потрібен для дат після 2025 року; для дат у межах 2017–2025 автоматично використовується `firms_points_2017_2025.csv.gz`).
- *(Опціонально, тільки якщо не використовуєте готовий `population_cache.csv`)*: `GEE_PROJECT`, `GEE_SERVICE_ACCOUNT`, `GEE_KEY_FILE`.
