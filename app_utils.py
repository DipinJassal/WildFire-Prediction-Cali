import json
import math
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from utils import COUNTY_ELEVATION

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
METRICS_F = MODEL_DIR / "metrics.json"

COUNTY_COORDS = {
    "Alameda County": (37.6469, -121.8887),
    "Alpine County": (38.5972, -119.8207),
    "Amador County": (38.4464, -120.6511),
    "Butte County": (39.6669, -121.6007),
    "Calaveras County": (38.2046, -120.5541),
    "Colusa County": (39.1775, -122.2370),
    "Contra Costa County": (37.9192, -121.9275),
    "Del Norte County": (41.7431, -123.8972),
    "El Dorado County": (38.7787, -120.5247),
    "Fresno County": (36.7582, -119.6493),
    "Glenn County": (39.5983, -122.3921),
    "Humboldt County": (40.6993, -123.8756),
    "Imperial County": (33.0395, -115.3653),
    "Inyo County": (36.5111, -117.4108),
    "Kern County": (35.3426, -118.7301),
    "Kings County": (36.0753, -119.8155),
    "Lake County": (39.0996, -122.7532),
    "Lassen County": (40.6736, -120.5943),
    "Los Angeles County": (34.3219, -118.2247),
    "Madera County": (37.2181, -119.7626),
    "Marin County": (38.0734, -122.7234),
    "Mariposa County": (37.5815, -119.9055),
    "Mendocino County": (39.4402, -123.3915),
    "Merced County": (37.1919, -120.7177),
    "Modoc County": (41.5898, -120.7250),
    "Mono County": (37.9391, -118.8869),
    "Monterey County": (36.2171, -121.2388),
    "Napa County": (38.5065, -122.3305),
    "Nevada County": (39.3014, -120.7688),
    "Orange County": (33.7031, -117.7609),
    "Placer County": (39.0634, -120.7177),
    "Plumas County": (40.0047, -120.8386),
    "Riverside County": (33.7437, -115.9939),
    "Sacramento County": (38.4493, -121.3443),
    "San Benito County": (36.6057, -121.0750),
    "San Bernardino County": (34.8414, -116.1785),
    "San Diego County": (33.0343, -116.7350),
    "San Francisco County": (37.7556, -122.4450),
    "San Joaquin County": (37.9348, -121.2714),
    "San Luis Obispo County": (35.3871, -120.4044),
    "San Mateo County": (37.4228, -122.3291),
    "Santa Barbara County": (34.6729, -120.0169),
    "Santa Clara County": (37.2318, -121.6951),
    "Santa Cruz County": (37.0562, -122.0018),
    "Shasta County": (40.7637, -122.0405),
    "Sierra County": (39.5804, -120.5161),
    "Siskiyou County": (41.5927, -122.5404),
    "Solano County": (38.2700, -121.9329),
    "Sonoma County": (38.5289, -122.8880),
    "Stanislaus County": (37.5591, -120.9977),
    "Sutter County": (39.0346, -121.6948),
    "Tehama County": (40.1257, -122.2340),
    "Trinity County": (40.6507, -123.1126),
    "Tulare County": (36.2202, -118.8005),
    "Tuolumne County": (38.0276, -119.9548),
    "Ventura County": (34.4561, -119.0836),
    "Yolo County": (38.6866, -121.9016),
    "Yuba County": (39.2690, -121.3513),
}


@st.cache_data
def load_metrics():
    if not METRICS_F.exists():
        return None
    with open(METRICS_F) as f:
        return json.load(f)


@st.cache_resource
def load_model(name: str):
    safe = name.replace(" ", "_").replace("+", "plus")
    path = MODEL_DIR / f"{safe}.pkl"
    if path.exists():
        return joblib.load(path)
    best = MODEL_DIR / "best_model.pkl"
    return joblib.load(best) if best.exists() else None


@st.cache_resource
def load_feature_cols():
    p = MODEL_DIR / "feature_cols.pkl"
    return joblib.load(p) if p.exists() else []


@st.cache_data
def load_geojson():
    p = MODEL_DIR / "california_counties.geojson"
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_shap_explainer(model_name: str):
    import shap

    model = load_model(model_name)
    if model is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        try:
            return shap.LinearExplainer(
                model,
                masker=shap.maskers.Independent(
                    pd.DataFrame(columns=load_feature_cols()), max_samples=100
                ),
            )
        except Exception:
            return None


def build_input_row(
    county,
    month,
    temp_max,
    temp_min,
    humidity,
    wind_speed,
    precip,
    drought,
    tmax_7d,
    hum_7d,
    tmax_14d,
    hum_14d,
    evapo=5.0,
    evapo_14d=5.0,
    prev_day_fire=0,
    prev2_day_fire=0,
    fire_7d=0,
    statewide_fires=5,
    base_rate=0.03,
    wind_dir=180,
) -> dict:
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    vpd = max(
        0,
        (1 - humidity / 100) * 0.6108 * math.exp(17.27 * temp_max / (temp_max + 237.3)),
    )
    fire_season = 1 if month in [6, 7, 8, 9, 10] else 0
    wind_dir_rad = wind_dir * math.pi / 180
    elevation = COUNTY_ELEVATION.get(county, 500)
    lat, lon = COUNTY_COORDS.get(county, (37.0, -120.0))

    # FFWI (Fosberg Fire Weather Index)
    wind_mph = wind_speed * 2.237
    temp_f = temp_max * 9 / 5 + 32
    h, t = humidity, temp_f
    if h < 10:
        emc = 0.03229 + 0.281073 * h - 0.000578 * t * h
    elif h <= 50:
        emc = 2.22749 + 0.160107 * h - 0.014784 * t
    else:
        emc = 21.0606 + 0.005565 * h**2 - 0.00035 * t * h - 0.483199 * h
    eta = max(0.0, min(1.0, emc / 30))
    ffwi = max(
        0.0,
        (wind_mph**2 + 1) ** 0.5 * (1 - 2 * eta + 1.5 * eta**2 - 0.5 * eta**3) / 0.3002,
    )

    return {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation": precip,
        "month": month,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_year": month * 30,
        "weekend_flag": 0,
        "fire_season_flag": fire_season,
        "lat": lat,
        "lon": lon,
        "temp_max_7d_ewma": tmax_7d,
        "humidity_7d_ewma": hum_7d,
        "temp_max_14d_ewma": tmax_14d,
        "humidity_14d_ewma": hum_14d,
        "temp_max_30d_ewma": temp_max,
        "humidity_30d_ewma": humidity,
        "temperature_anomaly": temp_max - temp_max,  # Approx
        "vpd": vpd,
        "drought_index": drought,
        "wind_speed_drought_interaction": wind_speed * drought,
        "temp_max_humidity_interaction": temp_max * humidity,
        "evapotranspiration": evapo,
        "evapotranspiration_14d_ewma": evapo_14d,
        "prev_day_fire": float(prev_day_fire),
        "prev2_day_fire": float(prev2_day_fire),
        "fire_7d_rolling": float(fire_7d),
        "statewide_fire_yesterday": float(statewide_fires),
        "historical_county_base_rate": float(base_rate),
        "elevation": float(elevation),
        "wind_dir_sin": math.sin(wind_dir_rad),
        "wind_dir_cos": math.cos(wind_dir_rad),
        "offshore_wind_flag": float((wind_dir <= 90) or (wind_dir >= 315)),
        "ffwi": ffwi,
    }


def batch_predict_all_counties(
    model,
    all_feat_cols,
    counties_list,
    month,
    temp_max,
    temp_min,
    humidity,
    wind_speed,
    precip,
    drought,
    wind_dir=180,
) -> pd.DataFrame:
    # We create unique rows for each county since spatial features (lat, lon, base rate, elevation) vary
    rows = []
    for county in counties_list:
        row = build_input_row(
            county,
            month,
            temp_max,
            temp_min,
            humidity,
            wind_speed,
            precip,
            drought,
            tmax_7d=temp_max,
            hum_7d=humidity,
            tmax_14d=temp_max,
            hum_14d=humidity,
            wind_dir=wind_dir,
        )
        rows.append(row)

    X = (
        pd.DataFrame(rows)
        .reindex(columns=all_feat_cols, fill_value=0)
        .fillna(0)
        .astype(float)
    )
    probas = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"county": counties_list, "fire_prob": probas})
