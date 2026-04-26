"""
Fetch missing counties from Open-Meteo, append to cache,
then rebuild splits and retrain models.

Safe to re-run any time — only fetches what's still missing.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE_F    = Path(__file__).parent / "dataset" / "weather_cache.parquet"
START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

COUNTY_COORDS = {
    "Alameda County":        (37.6469, -121.8887),
    "Alpine County":         (38.5972, -119.8207),
    "Amador County":         (38.4464, -120.6511),
    "Butte County":          (39.6669, -121.6007),
    "Calaveras County":      (38.2046, -120.5541),
    "Colusa County":         (39.1775, -122.2370),
    "Contra Costa County":   (37.9192, -121.9275),
    "Del Norte County":      (41.7431, -123.8972),
    "El Dorado County":      (38.7787, -120.5247),
    "Fresno County":         (36.7582, -119.6493),
    "Glenn County":          (39.5983, -122.3921),
    "Humboldt County":       (40.6993, -123.8756),
    "Imperial County":       (33.0395, -115.3653),
    "Inyo County":           (36.5111, -117.4108),
    "Kern County":           (35.3426, -118.7301),
    "Kings County":          (36.0753, -119.8155),
    "Lake County":           (39.0996, -122.7532),
    "Lassen County":         (40.6736, -120.5943),
    "Los Angeles County":    (34.3219, -118.2247),
    "Madera County":         (37.2181, -119.7626),
    "Marin County":          (38.0734, -122.7234),
    "Mariposa County":       (37.5815, -119.9055),
    "Mendocino County":      (39.4402, -123.3915),
    "Merced County":         (37.1919, -120.7177),
    "Modoc County":          (41.5898, -120.7250),
    "Mono County":           (37.9391, -118.8869),
    "Monterey County":       (36.2171, -121.2388),
    "Napa County":           (38.5065, -122.3305),
    "Nevada County":         (39.3014, -120.7688),
    "Orange County":         (33.7031, -117.7609),
    "Placer County":         (39.0634, -120.7177),
    "Plumas County":         (40.0047, -120.8386),
    "Riverside County":      (33.7437, -115.9939),
    "Sacramento County":     (38.4493, -121.3443),
    "San Benito County":     (36.6057, -121.0750),
    "San Bernardino County": (34.8414, -116.1785),
    "San Diego County":      (33.0343, -116.7350),
    "San Francisco County":  (37.7556, -122.4450),
    "San Joaquin County":    (37.9348, -121.2714),
    "San Luis Obispo County":(35.3871, -120.4044),
    "San Mateo County":      (37.4228, -122.3291),
    "Santa Barbara County":  (34.6729, -120.0169),
    "Santa Clara County":    (37.2318, -121.6951),
    "Santa Cruz County":     (37.0562, -122.0018),
    "Shasta County":         (40.7637, -122.0405),
    "Sierra County":         (39.5804, -120.5161),
    "Siskiyou County":       (41.5927, -122.5404),
    "Solano County":         (38.2700, -121.9329),
    "Sonoma County":         (38.5289, -122.8880),
    "Stanislaus County":     (37.5591, -120.9977),
    "Sutter County":         (39.0346, -121.6948),
    "Tehama County":         (40.1257, -122.2340),
    "Trinity County":        (40.6507, -123.1126),
    "Tulare County":         (36.2202, -118.8005),
    "Tuolumne County":       (38.0276, -119.9548),
    "Ventura County":        (34.4561, -119.0836),
    "Yolo County":           (38.6866, -121.9016),
    "Yuba County":           (39.2690, -121.3513),
}


def fetch_county(county: str, lat: float, lon: float,
                 start: str, end: str, delay: float = 5.0) -> pd.DataFrame | None:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"relative_humidity_2m_max,wind_speed_10m_max,precipitation_sum"
        f"&timezone=America%2FLos_Angeles"
    )
    backoff = delay
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            d = data["daily"]
            return pd.DataFrame({
                "county":        county,
                "date":          pd.to_datetime(d["time"]),
                "temp_max":      d["temperature_2m_max"],
                "temp_min":      d["temperature_2m_min"],
                "humidity":      d["relative_humidity_2m_max"],
                "wind_speed":    d["wind_speed_10m_max"],
                "precipitation": d["precipitation_sum"],
            })
        except Exception as e:
            if "429" in str(e):
                print(f"    rate-limited, waiting {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2       # exponential backoff: 5→10→20→40→80s
            else:
                print(f"    error: {e}")
                time.sleep(5)
    return None


def main():
    cached    = pd.read_parquet(CACHE_F) if CACHE_F.exists() else pd.DataFrame()
    cached_ok = set(cached["county"].unique()) if not cached.empty else set()
    missing   = {c: v for c, v in COUNTY_COORDS.items() if c not in cached_ok}

    if not missing:
        print("All 58 counties already cached — nothing to fetch.")
    else:
        print(f"Fetching {len(missing)} missing counties (5s gap, exponential backoff on 429)...\n")
        new_frames = []
        failed     = []

        for i, (county, (lat, lon)) in enumerate(missing.items(), 1):
            print(f"  [{i:2d}/{len(missing)}] {county} ...", end=" ", flush=True)
            df = fetch_county(county, lat, lon, START_DATE, END_DATE)
            if df is not None:
                new_frames.append(df)
                print(f"OK ({len(df)} days)")
            else:
                failed.append(county)
                print("FAILED (will retry next run)")
            time.sleep(5)   # 5s between every request

        if new_frames:
            updated = pd.concat([cached] + new_frames, ignore_index=True)
            updated.to_parquet(CACHE_F, index=False)
            total = updated["county"].nunique()
            print(f"\nCache updated: {total}/58 counties")
        else:
            print("\nNo new data fetched.")

        if failed:
            print(f"Still missing ({len(failed)}): {', '.join(failed)}")
            print("Re-run this script later to retry.")
            return

    # All 58 counties present — rebuild splits and retrain
    print("\nAll counties cached. Rebuilding splits...")
    import subprocess, sys
    subprocess.run([sys.executable, "rebuild_pipeline.py"], check=True)

    print("\nRetraining models...")
    subprocess.run([sys.executable, "train_models.py"], check=True)

    print("\nDone — app is ready with all 58 counties.")


if __name__ == "__main__":
    main()
