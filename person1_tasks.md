# Person 1: Data Collection & Assembly — Claude Code Instructions

## Context
I am building a Wildfire Risk Prediction project for my DATA 245 (Machine Learning) class at SJSU. I am using NASA FIRMS MODIS Collection 6.1 data for California from 2016-2025. The combined CSV file `firms_california_combined.csv` is already downloaded and filtered to type==0 and confidence>=80. It has ~103,000 fire detection rows.

## Dataset Details
- **File:** `firms_california_combined.csv`
- **Columns:** latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_t31, frp, daynight, type
- **Date range:** 2016-01-15 to 2025-12-29
- **All rows are type==0 (presumed vegetation fire) and confidence>=80**
- **Region:** California only (lat: 32.5-42.0, lon: -124.5 to -114.0)
- **Source documentation:** MODIS C6/C6.1 Active Fire Product User Guide, Page 39, Table 10 defines type 0 as "presumed vegetation fire"

## What I Need You To Do

### Task 1: Download California County Shapefile
- Download the California county boundary shapefile from US Census Bureau
- URL: https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_06_cousub_500k.zip
- If that URL doesn't work, try alternative sources or use the `geopandas.datasets` approach
- We need polygons for all 58 California counties

### Task 2: Spatial Join — Assign Each Fire Detection to a County
- Load `firms_california_combined.csv`
- Convert each row's (latitude, longitude) into a Point geometry
- Load the California county shapefile
- Perform a spatial join using `gpd.sjoin()` to assign each fire detection to a county
- After the join, keep the county name column (likely called `NAME` or `NAMELSAD`)
- Verify: print how many unique counties were matched, print value counts of top 10 counties by fire count
- Save result as `firms_with_counties.csv` — keep columns: latitude, longitude, brightness, acq_date, acq_time, satellite, confidence, bright_t31, frp, daynight, type, county

### Task 3: Build the Base Dataset with Non-Fire Days
This is the critical step. FIRMS only has rows where fires occurred. We need to create rows for days where NO fire occurred.

**Logic:**
1. Get the list of all 58 California county names from the shapefile
2. Generate every date from 2016-01-01 to 2025-12-31 using `pd.date_range()`
3. Create ALL combinations of (county, date) — this should be 58 counties × 3,653 days = ~211,874 rows
4. From `firms_with_counties.csv`, group by (county, acq_date) and aggregate:
   - `max_frp`: max of frp column
   - `max_brightness`: max of brightness column  
   - `fire_count`: count of detections
5. LEFT JOIN the all_combinations dataframe with the fire aggregated dataframe on (county, date)
6. Create `fire_label` column: 1 if max_frp is not NaN (fire detected), 0 if NaN (no fire)
7. Fill NaN values in max_frp, max_brightness, fire_count with 0

**Final base_dataset.csv should have these columns:**
- county (string)
- date (datetime)
- fire_label (0 or 1)
- max_frp (float, 0 for non-fire days)
- max_brightness (float, 0 for non-fire days)
- fire_count (int, 0 for non-fire days)

### Task 4: Validation & Summary Statistics
After building base_dataset.csv, print the following:
1. Total rows (should be ~212,000)
2. Number and percentage of fire_label=1 vs fire_label=0
3. Fire counts per year
4. Top 10 counties by total fire days
5. Monthly distribution of fire days (which months have most fires)
6. Verify January 2025 LA fires are captured: check Los Angeles county fire detections in Jan 2025
7. Basic stats of max_frp for fire days only (mean, median, max)
8. Confirm 58 unique counties exist in the dataset

### Task 5: Save Everything
- Save `firms_with_counties.csv` (fire detections with county assigned)
- Save `base_dataset.csv` (the full ~212K row dataset with fire and non-fire days)
- Save a `data_summary.txt` with all the validation stats from Task 4

## Libraries Needed
```
pip install pandas geopandas shapely
```

## Important Notes
- The spatial join coordinate reference system should be EPSG:4326 for both the fire points and the county polygons
- Some fire detections may fall outside county boundaries (in ocean or on exact borders) — it's okay to drop these, should be less than 1%
- The county name column in Census shapefiles is usually `NAME` — check and use the correct one
- For the all_combinations step, use `itertools.product` or `pd.MultiIndex.from_product`
- acq_date in FIRMS is a string like "2016-01-15" — convert to datetime before joining
- The base_dataset.csv is what Person 2 will use to add weather features, so make sure it's clean

## File Structure
```
project/
├── firms_california_combined.csv    (input - already have this)
├── ca_counties/                     (downloaded shapefile)
├── firms_with_counties.csv          (output - fires with county names)
├── base_dataset.csv                 (output - full dataset with non-fire days)
└── data_summary.txt                 (output - validation stats)
```
