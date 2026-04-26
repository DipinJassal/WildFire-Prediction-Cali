"""
Full Pipeline Rebuild — 2016-2025
Fetches weather for all 58 counties, merges with fire data,
engineers features, and creates new train/val/test splits.

Split: train=2016-2022, val=2023, test=2024-2025
"""

import json
import time
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import COUNTY_ELEVATION

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATA_DIR     = Path("/Users/dipinjassal/Downloads/Data_Split")
BASE_CSV     = ROOT / "dataset" / "base_dataset.csv"
CACHE_F      = ROOT / "dataset" / "weather_cache.parquet"
WIND_CACHE_F = ROOT / "dataset" / "wind_cache.parquet"

START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

# ── County centroids (from shapefile) ─────────────────────────────────────────
COUNTY_COORDS = {
    "Alameda County":       (37.6469, -121.8887),
    "Alpine County":        (38.5972, -119.8207),
    "Amador County":        (38.4464, -120.6511),
    "Butte County":         (39.6669, -121.6007),
    "Calaveras County":     (38.2046, -120.5541),
    "Colusa County":        (39.1775, -122.2370),
    "Contra Costa County":  (37.9192, -121.9275),
    "Del Norte County":     (41.7431, -123.8972),
    "El Dorado County":     (38.7787, -120.5247),
    "Fresno County":        (36.7582, -119.6493),
    "Glenn County":         (39.5983, -122.3921),
    "Humboldt County":      (40.6993, -123.8756),
    "Imperial County":      (33.0395, -115.3653),
    "Inyo County":          (36.5111, -117.4108),
    "Kern County":          (35.3426, -118.7301),
    "Kings County":         (36.0753, -119.8155),
    "Lake County":          (39.0996, -122.7532),
    "Lassen County":        (40.6736, -120.5943),
    "Los Angeles County":   (34.3219, -118.2247),
    "Madera County":        (37.2181, -119.7626),
    "Marin County":         (38.0734, -122.7234),
    "Mariposa County":      (37.5815, -119.9055),
    "Mendocino County":     (39.4402, -123.3915),
    "Merced County":        (37.1919, -120.7177),
    "Modoc County":         (41.5898, -120.7250),
    "Mono County":          (37.9391, -118.8869),
    "Monterey County":      (36.2171, -121.2388),
    "Napa County":          (38.5065, -122.3305),
    "Nevada County":        (39.3014, -120.7688),
    "Orange County":        (33.7031, -117.7609),
    "Placer County":        (39.0634, -120.7177),
    "Plumas County":        (40.0047, -120.8386),
    "Riverside County":     (33.7437, -115.9939),
    "Sacramento County":    (38.4493, -121.3443),
    "San Benito County":    (36.6057, -121.0750),
    "San Bernardino County":(34.8414, -116.1785),
    "San Diego County":     (33.0343, -116.7350),
    "San Francisco County": (37.7556, -122.4450),
    "San Joaquin County":   (37.9348, -121.2714),
    "San Luis Obispo County":(35.3871, -120.4044),
    "San Mateo County":     (37.4228, -122.3291),
    "Santa Barbara County": (34.6729, -120.0169),
    "Santa Clara County":   (37.2318, -121.6951),
    "Santa Cruz County":    (37.0562, -122.0018),
    "Shasta County":        (40.7637, -122.0405),
    "Sierra County":        (39.5804, -120.5161),
    "Siskiyou County":      (41.5927, -122.5404),
    "Solano County":        (38.2700, -121.9329),
    "Sonoma County":        (38.5289, -122.8880),
    "Stanislaus County":    (37.5591, -120.9977),
    "Sutter County":        (39.0346, -121.6948),
    "Tehama County":        (40.1257, -122.2340),
    "Trinity County":       (40.6507, -123.1126),
    "Tulare County":        (36.2202, -118.8005),
    "Tuolumne County":      (38.0276, -119.9548),
    "Ventura County":       (34.4561, -119.0836),
    "Yolo County":          (38.6866, -121.9016),
    "Yuba County":          (39.2690, -121.3513),
}


# ── Weather fetch ──────────────────────────────────────────────────────────────
def fetch_weather_county(county: str, lat: float, lon: float,
                          start: str, end: str, retries: int = 3) -> pd.DataFrame:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"relative_humidity_2m_max,wind_speed_10m_max,precipitation_sum"
        f"&timezone=America%2FLos_Angeles"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            d = data["daily"]
            df = pd.DataFrame({
                "county":        county,
                "date":          pd.to_datetime(d["time"]),
                "temp_max":      d["temperature_2m_max"],
                "temp_min":      d["temperature_2m_min"],
                "humidity":      d["relative_humidity_2m_max"],
                "wind_speed":    d["wind_speed_10m_max"],
                "precipitation": d["precipitation_sum"],
            })
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: failed for {county}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_wind_direction() -> pd.DataFrame:
    """Fetch wind_direction_10m_dominant for all counties into a separate cache."""
    cached    = pd.read_parquet(WIND_CACHE_F) if WIND_CACHE_F.exists() else pd.DataFrame()
    cached_ok = set(cached["county"].unique()) if not cached.empty else set()
    missing   = {c: v for c, v in COUNTY_COORDS.items() if c not in cached_ok}

    if not missing:
        print(f"Wind direction cached for all {len(COUNTY_COORDS)} counties")
        return cached

    print(f"Fetching wind direction for {len(missing)} counties...")
    new_frames = []
    for i, (county, (lat, lon)) in enumerate(missing.items(), 1):
        print(f"  [{i:2d}/{len(missing)}] {county}", end=" ... ", flush=True)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&daily=wind_direction_10m_dominant"
            f"&timezone=America%2FLos_Angeles"
        )
        backoff = 5.0
        success = False
        for _ in range(6):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read())
                d = data["daily"]
                new_frames.append(pd.DataFrame({
                    "county":         county,
                    "date":           pd.to_datetime(d["time"]),
                    "wind_direction": d["wind_direction_10m_dominant"],
                }))
                print("OK")
                success = True
                break
            except Exception as e:
                if "429" in str(e):
                    print(f"rate-limited {backoff:.0f}s...", end=" ", flush=True)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                else:
                    print(f"error: {e}")
                    break
        if not success:
            print("FAILED")
        time.sleep(3.0)

    if new_frames:
        updated = pd.concat([cached] + new_frames, ignore_index=True)
        updated.to_parquet(WIND_CACHE_F, index=False)
        print(f"Wind cache: {updated['county'].nunique()}/58 counties")
        return updated
    return cached


def fetch_all_weather() -> pd.DataFrame:
    # Load existing cache if present
    cached = pd.read_parquet(CACHE_F) if CACHE_F.exists() else pd.DataFrame()
    cached_counties = set(cached["county"].unique()) if not cached.empty else set()
    missing = {c: v for c, v in COUNTY_COORDS.items() if c not in cached_counties}

    if not missing:
        print(f"All {len(COUNTY_COORDS)} counties cached — loading from {CACHE_F}")
        return cached

    print(f"Fetching {len(missing)} missing counties ({START_DATE} → {END_DATE})...")
    frames = [cached] if not cached.empty else []
    for i, (county, (lat, lon)) in enumerate(missing.items(), 1):
        print(f"  [{i:2d}/{len(missing)}] {county}", end=" ... ", flush=True)
        df = fetch_weather_county(county, lat, lon, START_DATE, END_DATE)
        if not df.empty:
            frames.append(df)
            print(f"OK ({len(df)} days)")
        else:
            print("SKIPPED")
        time.sleep(2.0)   # respect rate limits

    weather = pd.concat(frames, ignore_index=True)
    weather.to_parquet(CACHE_F, index=False)
    print(f"Weather cached: {weather.shape} ({weather['county'].nunique()} counties)")
    return weather


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["county", "date"]).reset_index(drop=True)

    # ── Temporal
    df["month"]           = df["date"].dt.month
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year"]     = df["date"].dt.dayofyear
    df["weekend_flag"]    = (df["date"].dt.dayofweek >= 5).astype(int)
    df["fire_season_flag"]= df["month"].isin([6, 7, 8, 9, 10]).astype(int)

    # ── Rolling weather (per county, no leakage: shift then roll)
    g = df.groupby("county")
    for days in [7, 14, 30]:
        df[f"temp_max_{days}d_rolling_mean"] = (
            g["temp_max"].transform(lambda x: x.shift(1).rolling(days, min_periods=1).mean())
        )
        df[f"humidity_{days}d_rolling_mean"] = (
            g["humidity"].transform(lambda x: x.shift(1).rolling(days, min_periods=1).mean())
        )

    # ── Temperature anomaly (vs 30-day rolling mean)
    df["temperature_anomaly"] = df["temp_max"] - df["temp_max_30d_rolling_mean"]

    # ── Vapour Pressure Deficit (kPa)
    df["vpd"] = (
        (1 - df["humidity"] / 100)
        * 0.6108
        * np.exp(17.27 * df["temp_max"] / (df["temp_max"] + 237.3))
    ).clip(lower=0)

    # ── Drought index: rolling 30-day precipitation deficit normalised 0-4
    precip_30d = g["precipitation"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).sum()
    )
    scaler = MinMaxScaler(feature_range=(0, 4))
    df["drought_index"] = 4 - scaler.fit_transform(precip_30d.values.reshape(-1, 1)).flatten()

    # ── Interaction features
    df["wind_speed_drought_interaction"]  = df["wind_speed"]  * df["drought_index"]
    df["temp_max_humidity_interaction"]   = df["temp_max"]    * df["humidity"]

    # ── Lag fire features (no leakage — shift within county)
    df["prev_day_fire"]   = g["fire_label"].transform(lambda x: x.shift(1)).fillna(0)
    df["prev2_day_fire"]  = g["fire_label"].transform(lambda x: x.shift(2)).fillna(0)
    df["fire_7d_rolling"] = g["fire_label"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0)

    # ── Elevation (static per county — higher elevation = different fire regime)
    df["elevation"] = df["county"].map(COUNTY_ELEVATION).fillna(500)

    # ── Wind direction (cyclical encoding — present only when fetched)
    if "wind_direction" in df.columns:
        df["wind_direction"] = df["wind_direction"].fillna(180)
        rad = df["wind_direction"] * np.pi / 180
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
        # Offshore/Santa Ana flag: wind from NE quadrant (0-90°) is dangerous for CA
        df["offshore_wind_flag"] = (
            (df["wind_direction"] <= 90) | (df["wind_direction"] >= 315)
        ).astype(int)

    # ── Fosberg Fire Weather Index (FFWI) — operational fire weather metric
    wind_mph = df["wind_speed"] * 2.237
    temp_f   = df["temp_max"] * 9 / 5 + 32
    h, t     = df["humidity"], temp_f
    emc = np.where(h < 10,  0.03229 + 0.281073 * h - 0.000578 * t * h,
          np.where(h <= 50, 2.22749  + 0.160107 * h - 0.014784 * t,
                             21.0606  + 0.005565 * h**2 - 0.00035 * t * h - 0.483199 * h))
    eta = np.clip(emc / 30, 0, 1)
    df["ffwi"] = np.maximum(
        (wind_mph**2 + 1)**0.5 * (1 - 2*eta + 1.5*eta**2 - 0.5*eta**3) / 0.3002, 0
    )

    # ── Handle any remaining NaNs in weather cols
    weather_cols = ["temp_max", "temp_min", "humidity", "wind_speed", "precipitation"]
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())

    # ── County one-hot
    dummies = pd.get_dummies(df["county"], prefix="county")
    df = pd.concat([df, dummies], axis=1)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Weather
    weather = fetch_all_weather()
    print(f"\nWeather shape: {weather.shape}")

    # 1b. Wind direction (separate cache, merged in)
    wind_df = fetch_wind_direction()
    if not wind_df.empty:
        weather = weather.merge(
            wind_df[["county", "date", "wind_direction"]],
            on=["county", "date"], how="left"
        )
        print(f"Wind direction merged — coverage: {weather['wind_direction'].notna().mean():.1%}")

    # 2. Fire labels
    fire = pd.read_csv(BASE_CSV, parse_dates=["date"])
    print(f"Fire data shape: {fire.shape}")
    print(f"Fire date range: {fire['date'].min().date()} → {fire['date'].max().date()}")

    # 3. Merge
    merged = weather.merge(
        fire[["county", "date", "fire_label", "max_frp", "max_brightness", "fire_count"]],
        on=["county", "date"], how="left"
    )
    merged["fire_label"]    = merged["fire_label"].fillna(0)
    merged["max_frp"]       = merged["max_frp"].fillna(0)
    merged["max_brightness"]= merged["max_brightness"].fillna(0)
    merged["fire_count"]    = merged["fire_count"].fillna(0)
    print(f"Merged shape: {merged.shape}")

    # 4. Feature engineering
    print("\nEngineering features...")
    df = engineer_features(merged)
    print(f"After feature engineering: {df.shape}")

    # 5. Fire rate check
    by_year = df.groupby(df["date"].dt.year)["fire_label"].agg(["sum", "count"])
    by_year["rate"] = (by_year["sum"] / by_year["count"] * 100).round(2)
    print("\nFire rate by year:")
    print(by_year.rename(columns={"sum": "fire_days", "count": "total_days", "rate": "fire_%"}).to_string())

    # 6. Drop non-feature columns before saving
    drop_cols = ["county", "date"]
    county_dummies = [c for c in df.columns if c.startswith("county_")]

    # Keep date for splitting, drop county string
    df_save = df.drop(columns=["county"])

    # 7. Time-based split
    year = df["date"].dt.year
    train = df_save[year <= 2022].copy()
    val   = df_save[(year == 2023)].copy()
    test  = df_save[(year >= 2024)].copy()

    # Drop date column after splitting
    train = train.drop(columns=["date"])
    val   = val.drop(columns=["date"])
    test  = test.drop(columns=["date"])

    print(f"\nSplit sizes:")
    print(f"  Train (2016-2022): {len(train):,} rows | fire rate: {train['fire_label'].mean():.3%}")
    print(f"  Val   (2023):      {len(val):,} rows | fire rate: {val['fire_label'].mean():.3%}")
    print(f"  Test  (2024-2025): {len(test):,} rows | fire rate: {test['fire_label'].mean():.3%}")

    # 8. SMOTE on training set
    print("\nApplying SMOTE to training set...")
    from imblearn.over_sampling import SMOTE

    leakage = ["max_frp", "max_brightness", "fire_count"]
    feat_cols = [c for c in train.columns if c not in ["fire_label"] + leakage]
    X_tr = train[feat_cols].astype(float)
    y_tr = train["fire_label"]

    # Impute any remaining NaNs with column median before SMOTE
    nan_cols = X_tr.columns[X_tr.isna().any()].tolist()
    if nan_cols:
        print(f"  Imputing NaNs in {len(nan_cols)} columns before SMOTE")
        X_tr = X_tr.fillna(X_tr.median())

    sm = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.2)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    smote_df = pd.DataFrame(X_sm, columns=feat_cols)
    smote_df["fire_label"] = y_sm
    print(f"  SMOTE: {len(X_tr):,} → {len(X_sm):,} rows")

    # 9. Save
    print("\nSaving splits...")
    train.to_csv(DATA_DIR / "train.csv", index=False)
    val.to_csv(DATA_DIR / "val.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    smote_df.to_csv(DATA_DIR / "train_features_smote.csv", index=False)
    print(f"  Saved to {DATA_DIR}")

    print("\nDone. Run `python train_models.py` next.")


if __name__ == "__main__":
    main()
