"""
Wildfire Risk Prediction — Person 1 Tasks 2–5
Builds firms_with_counties.csv and base_dataset.csv from FIRMS MODIS data.
"""

import pandas as pd
import geopandas as gpd
import itertools

# ── Paths ──────────────────────────────────────────────────────────────────────
SHAPEFILE = "cb_2020_06_cousub_500k/cb_2020_06_cousub_500k.shp"
FIRMS_CSV = "firms_california_combined.csv"
OUT_FIRMS = "firms_with_counties.csv"
OUT_BASE = "base_dataset.csv"
OUT_SUMMARY = "data_summary.txt"

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Spatial Join — assign each fire detection to a county
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 2: Spatial Join")
print("=" * 60)

# Load county subdivisions and dissolve to county-level polygons
cousub = gpd.read_file(SHAPEFILE)
counties = (
    cousub.dissolve(by="COUNTYFP", as_index=False)[
        ["COUNTYFP", "NAMELSADCO", "geometry"]
    ]
    .rename(columns={"NAMELSADCO": "county"})
    .to_crs("EPSG:4326")  # reproject to WGS-84 to match fire lat/lon
)
print(f"County polygons loaded: {len(counties)} counties")

# Load fire detections
fires = pd.read_csv(FIRMS_CSV)
print(f"Fire detections loaded: {len(fires):,} rows")

# Convert to GeoDataFrame (lat/lon → Point, EPSG:4326)
gdf_fires = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(fires["longitude"], fires["latitude"]),
    crs="EPSG:4326",
)

# Spatial join: each point gets the county it falls in
joined = gpd.sjoin(
    gdf_fires, counties[["county", "geometry"]], how="left", predicate="within"
)

# For points that didn't match (gaps between subdivision polygons),
# use nearest-neighbor to assign the closest county
unmatched_mask = joined["county"].isna()
unmatched_count = unmatched_mask.sum()
print(
    f"Unmatched after 'within' join: {unmatched_count} ({unmatched_count/len(joined)*100:.2f}%)"
)

if unmatched_count > 0:
    # Pull the original GeoDataFrame rows for unmatched points
    unmatched_gdf = gdf_fires.loc[joined[unmatched_mask].index]
    # Reproject to metric CRS (California Albers) for accurate nearest-neighbor
    PROJ = "EPSG:3310"
    nearest = gpd.sjoin_nearest(
        unmatched_gdf.to_crs(PROJ),
        counties[["county", "geometry"]].to_crs(PROJ),
        how="left",
    )
    # Drop index_right if it crept in, keep county
    nearest = nearest[["county"]]
    joined.loc[unmatched_mask, "county"] = nearest["county"].values
    still_missing = joined["county"].isna().sum()
    print(f"Recovered via nearest-neighbor: {unmatched_count - still_missing}")
    print(f"Still unmatched (truly offshore): {still_missing}")
    joined = joined.dropna(subset=["county"])

# Keep required columns only
keep_cols = [
    "latitude",
    "longitude",
    "brightness",
    "acq_date",
    "acq_time",
    "satellite",
    "confidence",
    "bright_t31",
    "frp",
    "daynight",
    "type",
    "county",
]
firms_with_counties = joined[keep_cols].reset_index(drop=True)

print(f"\nUnique counties matched: {firms_with_counties['county'].nunique()}")
print("\nTop 10 counties by fire detection count:")
print(firms_with_counties["county"].value_counts().head(10).to_string())

firms_with_counties.to_csv(OUT_FIRMS, index=False)
print(f"\nSaved → {OUT_FIRMS}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: Build base dataset with non-fire days
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 3: Building Base Dataset (fire + non-fire days)")
print("=" * 60)

# All 58 county names
all_counties = sorted(counties["county"].tolist())
print(f"Counties: {len(all_counties)}")

# All dates 2016-01-01 to 2025-12-31
all_dates = pd.date_range("2016-01-01", "2025-12-31", freq="D")
print(f"Dates: {len(all_dates)} days")

# All (county, date) combinations
combos = pd.DataFrame(
    list(itertools.product(all_counties, all_dates)),
    columns=["county", "date"],
)
print(f"All combinations: {len(combos):,} rows (expected ~{58 * len(all_dates):,})")

# Aggregate fire detections by (county, acq_date)
firms_with_counties["acq_date"] = pd.to_datetime(firms_with_counties["acq_date"])

fire_agg = (
    firms_with_counties.groupby(["county", "acq_date"])
    .agg(
        max_frp=("frp", "max"),
        max_brightness=("brightness", "max"),
        fire_count=("frp", "count"),
    )
    .reset_index()
    .rename(columns={"acq_date": "date"})
)

# Left join: all_combinations ← fire aggregates
base = combos.merge(fire_agg, on=["county", "date"], how="left")

# fire_label: 1 if there was a fire detection, 0 otherwise
base["fire_label"] = base["max_frp"].notna().astype(int)

# Fill NaN → 0 for numeric columns
base[["max_frp", "max_brightness", "fire_count"]] = base[
    ["max_frp", "max_brightness", "fire_count"]
].fillna(0)
base["fire_count"] = base["fire_count"].astype(int)

# Ensure column order
base = base[["county", "date", "fire_label", "max_frp", "max_brightness", "fire_count"]]

base.to_csv(OUT_BASE, index=False)
print(f"\nSaved → {OUT_BASE}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: Validation & Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 4: Validation & Summary Statistics")
print("=" * 60)

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


log("=" * 60)
log("WILDFIRE RISK DATASET — SUMMARY STATISTICS")
log("=" * 60)

# 1. Total rows
log(f"\n1. Total rows: {len(base):,}")

# 2. Fire label distribution
fire_counts = base["fire_label"].value_counts()
total = len(base)
log("\n2. Fire label distribution:")
log(
    f"   fire_label=1 (fire day):    {fire_counts.get(1, 0):>7,}  ({fire_counts.get(1, 0)/total*100:.2f}%)"
)
log(
    f"   fire_label=0 (no fire day): {fire_counts.get(0, 0):>7,}  ({fire_counts.get(0, 0)/total*100:.2f}%)"
)

# 3. Fire counts per year
log("\n3. Fire days per year:")
base["year"] = pd.to_datetime(base["date"]).dt.year
yearly = base[base["fire_label"] == 1].groupby("year").size()
for yr, cnt in yearly.items():
    log(f"   {yr}: {cnt:,}")

# 4. Top 10 counties by total fire days
log("\n4. Top 10 counties by total fire days:")
county_fire = (
    base[base["fire_label"] == 1].groupby("county").size().sort_values(ascending=False)
)
for county, cnt in county_fire.head(10).items():
    log(f"   {county:<30} {cnt:,}")

# 5. Monthly distribution of fire days
log("\n5. Monthly distribution of fire days:")
base["month"] = pd.to_datetime(base["date"]).dt.month
monthly = base[base["fire_label"] == 1].groupby("month").size()
month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
for m, cnt in monthly.items():
    log(f"   {month_names[m]}: {cnt:,}")

# 6. January 2025 LA fires check
log("\n6. Los Angeles County fire detections — January 2025:")
la_jan25 = base[
    (base["county"] == "Los Angeles County")
    & (pd.to_datetime(base["date"]).dt.year == 2025)
    & (pd.to_datetime(base["date"]).dt.month == 1)
]
la_fire_days = la_jan25[la_jan25["fire_label"] == 1]
log(f"   Total Jan 2025 days checked: {len(la_jan25)}")
log(f"   Fire days detected:          {len(la_fire_days)}")
if len(la_fire_days) > 0:
    log(f"   Fire dates: {sorted(la_fire_days['date'].astype(str).tolist())}")

# 7. max_frp stats for fire days
frp_fire = base.loc[base["fire_label"] == 1, "max_frp"]
log("\n7. max_frp stats (fire days only):")
log(f"   Mean:   {frp_fire.mean():.2f}")
log(f"   Median: {frp_fire.median():.2f}")
log(f"   Max:    {frp_fire.max():.2f}")

# 8. Unique county count
log(f"\n8. Unique counties in dataset: {base['county'].nunique()}")

# Save summary
with open(OUT_SUMMARY, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {OUT_SUMMARY}")
log("\nAll tasks complete.")
