# Wildfire Risk Prediction — California

Predicting daily wildfire risk across all 58 California counties using NASA FIRMS satellite fire detection data (2016–2025). Built for DATA 245 (Machine Learning) at San Jose State University.

## Project Overview

The model predicts whether a wildfire will occur in a given California county on a given day (`fire_label`: 0 or 1). The pipeline starts from raw NASA MODIS active fire detections and builds a structured panel dataset covering every county × every day over a 10-year window, then enriches it with weather features for ML training.

## Dataset

**Source:** NASA FIRMS MODIS Collection 6.1 Active Fire Product  
**Region:** California (lat 32.5–42.0, lon -124.5 to -114.0)  
**Date range:** 2016-01-01 to 2025-12-31  
**Filter:** `type == 0` (presumed vegetation fire) and `confidence >= 80`

**County boundaries:** US Census Bureau TIGER/Line 2020 county subdivision shapefile  
Download: `https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_06_cousub_500k.zip`  
Place extracted folder at `dataset/cb_2020_06_cousub_500k/` before running the pipeline.

## Repository Structure

```
WildFire-Prediction-Cali/
├── dataset/
│   ├── data.py                      # Combines and filters raw yearly FIRMS CSVs
│   ├── build_dataset.py             # Spatial join + base dataset construction (Tasks 2–5)
│   ├── firms_california_combined.csv  # Filtered FIRMS fire detections (103,186 rows)
│   ├── firms_with_counties.csv      # Fire detections with county assigned (103,186 rows)
│   ├── base_dataset.csv             # Full ML-ready panel dataset (211,874 rows)
│   └── data_summary.txt             # Validation statistics
└── person1_tasks.md                 # Data collection & assembly task spec
```

## Pipeline

### Step 1 — Combine raw data (`data.py`)
Reads yearly FIRMS CSVs, concatenates them, filters to `type==0` and `confidence>=80`, and saves `firms_california_combined.csv`.

### Step 2 — Build dataset (`build_dataset.py`)

**Spatial join:** Converts each fire detection's lat/lon to a point geometry and joins to county polygons. Points that fall in shapefile gaps (county subdivision boundaries) are recovered via nearest-neighbor assignment — 0 records dropped.

**Base dataset:** Creates all 58 county × 3,653 day combinations (~211,874 rows), left-joins aggregated fire detections, and labels each row:
- `fire_label = 1` if any fire detection occurred in that county on that day
- `fire_label = 0` otherwise

## Output Files

| File | Rows | Description |
|---|---|---|
| `firms_california_combined.csv` | 103,186 | Filtered FIRMS detections |
| `firms_with_counties.csv` | 103,186 | Detections + county name |
| `base_dataset.csv` | 211,874 | Full panel dataset for ML |
| `data_summary.txt` | — | Validation stats |

### base_dataset.csv columns

| Column | Type | Description |
|---|---|---|
| `county` | string | California county name |
| `date` | datetime | Date (2016-01-01 to 2025-12-31) |
| `fire_label` | int (0/1) | Target variable — 1 if fire detected |
| `max_frp` | float | Max Fire Radiative Power (MW) that day, 0 if no fire |
| `max_brightness` | float | Max brightness temperature (K) that day, 0 if no fire |
| `fire_count` | int | Number of MODIS detections that day, 0 if no fire |

## Key Stats

- **Total rows:** 211,874
- **Fire days:** 8,637 (4.08%)
- **Non-fire days:** 203,237 (95.92%)
- **Peak fire month:** August
- **Top county:** Tulare County (521 fire days over 10 years)
- **LA January 2025 fires:** 10 fire days captured (Jan 7–23)

## Requirements

```
pip install pandas geopandas shapely
```

## Contributors

- Dipin Jassal
