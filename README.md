# Wildfire Risk Prediction — California

Predicting daily wildfire risk across all 58 California counties using NASA FIRMS satellite fire detection data (2016–2025). Built for DATA 245 (Machine Learning) at San Jose State University.

## Project Overview

The model predicts whether a wildfire will occur in a given California county on a given day (`fire_label`: 0 or 1). The pipeline starts from raw NASA MODIS active fire detections and builds a structured panel dataset covering every county × every day over a 10-year window, then enriches it with weather features for ML training.

## Dataset

### 1) NASA FIRMS fire data

**Source:** NASA FIRMS MODIS Collection 6.1 Active Fire Product  
**Region:** California (lat 32.5–42.0, lon -124.5 to -114.0)  
**Date range:** 2016-01-01 to 2025-12-31  
**Filter:** `type == 0` (presumed vegetation fire) and `confidence >= 80`

### 2) California county boundaries

**County boundaries:** US Census Bureau TIGER/Line 2020 county subdivision shapefile  
Download: `https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_06_cousub_500k.zip`  
Place extracted folder at `dataset/cb_2020_06_cousub_500k/` before running the pipeline.

### 3) Open-Meteo historical weather data

- Source: Open-Meteo Historical API
- Coverage: daily weather for county centroids from **2010-01-01 to 2020-12-31**
- Weather fields collected:
  - `temp_max`
  - `temp_min`
  - `humidity`
  - `wind_speed`
  - `precipitation`
- Output from Phase 2:
  - `weather_by_county.csv`
  - `dataset_with_all_features.csv`

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

## Phase 2 — Weather data & feature engineering (`Weather_Data_and_Feature_Engineering.ipynb`)

Creates a daily county-level weather table from Open-Meteo and engineers features for modeling.

**Step 2A — County centroids**
- Loads the California counties shapefile and computes 58 county centroids (lat/lon) for weather queries.

**Step 2B — Download daily Open-Meteo weather (2010–2020)**
- Queries Open-Meteo Historical API for each county centroid (daily):
  - `temperature_2m_max` → `temp_max`
  - `temperature_2m_min` → `temp_min`
  - `windspeed_10m_max` → `wind_speed`
  - `precipitation_sum` → `precipitation`
  - `dewpoint_2m_mean` (used to compute humidity)
- Computes relative humidity from dewpoint (Tetens approximation).
- Writes `weather_by_county.csv` (233,044 rows = 58 counties × 4,018 days).

**Step 2C — Merge + engineer features**
- Merges `weather_by_county.csv` with `base_dataset.csv` on (`county`, `date`).
- Engineers the following features:
  - **Calendar**: `month`, `month_sin`, `month_cos`, `day_of_year`, `weekend_flag`, `fire_season_flag` (Jun–Nov)
  - **Drought index**: consecutive dry days where `precipitation < 0.1` (resets on rain day)
  - **Rolling means** (per county, `min_periods=1`):  
    `temp_max_{7,14,30}d_rolling_mean`, `humidity_{7,14,30}d_rolling_mean`
  - **Temperature anomaly**: `temperature_anomaly = temp_max - temp_max_30d_rolling_mean`
  - **VPD**: `vpd = (1 - humidity/100) * 0.6108 * exp(17.27*temp_max/(temp_max + 237.3))`
  - **Interactions**: `wind_speed_drought_interaction`, `temp_max_humidity_interaction`
- Writes `dataset_with_all_features.csv` (28 columns total in the notebook output).

### Key Stats

- **Panel size:** ~233k county-day rows (58 counties × 2010–2020 daily dates)

## Phase 3 — Exploratory Data Analysis & Preprocessing (EDA & Data Prep)
This phase analyzes the dataset and prepares it for machine learning models by handling missing values, outliers, encoding, and dataset splitting.

---

### Exploratory Data Analysis (EDA)

Performed comprehensive analysis with **15+ visualizations**:

- **Class Distribution:** Highly imbalanced dataset (~2–4% fire days)
- **Seasonality:** Fire activity peaks during **June–October**
- **Yearly Trends:** Significant variation with spikes (e.g., 2018, 2020)
- **Top Counties:** Fire activity concentrated in specific regions (e.g., Tulare, Fresno)
- **Correlation Analysis:**  
  - Temperature → positive correlation with fire risk  
  - Humidity → negative correlation  
  - VPD & drought index → strong indicators
- **Feature Distributions:** Fire days occur under **hotter, drier, and windier conditions**
- **Rolling Features:** Sustained heat/dryness strongly linked to fires
- **Geospatial Map:** County-level wildfire activity visualization

Notebook: `Exploratory_Data_Analysis.ipynb`

---

### Preprocessing Pipeline

#### 1. Missing Values
- Applied **KNN Imputation (k=5)** to weather and engineered features

#### 2. Outlier Handling
- Applied **IQR capping** to:
  - `temp_max`
  - `wind_speed`

#### 3. Time-Based Split (No Random Split)
- **Train:** 2010–2017  
- **Validation:** 2018–2019  
- **Test:** 2020  

#### 4. Encoding
- One-hot encoding applied to `county`
- Ensured consistent feature space across splits

#### 5. Feature Handling for Modeling
- Dropped leakage features for classification models:
  - `max_frp`
  - `max_brightness`
  - `fire_count`
- These features are **retained in datasets** for regression tasks

#### 6. Class Imbalance Handling
- Applied **SMOTE (training set only)** to balance classes

#### 7. Scaling
- `StandardScaler` fitted on training data
- Saved as `scaler.pkl`
- Applied later during model training (for Logistic Regression and SVM)

---

### Output Files

| File | Description |
|------|------------|
| `train.csv` | Full training dataset (includes `fire_label` and `max_frp`) |
| `val.csv` | Validation dataset |
| `test.csv` | Test dataset |
| `train_smote.csv` | Balanced training dataset (classification only) |
| `scaler.pkl` | Saved scaler (fit on training data only) |

---

### Modeling Compatibility

- **Classification Task:** Predict `fire_label`
- **Regression Task:** Predict `max_frp` (only for fire days)

Note:
- `max_frp` is preserved in datasets to support regression modeling
- SMOTE is applied **only on training data**
- No data leakage introduced in classification features

## Contributors

- Dipin Jassal
- Huu Nguyen
- Samruddhi Chitnis
