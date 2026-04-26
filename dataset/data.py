import pandas as pd
import glob

files = glob.glob("20*.csv")
df = pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)

print(f"Total raw records: {len(df)}")
df = df[(df["type"] == 0) & (df["confidence"] >= 80)]
print(f"After filtering: {len(df)} records")
df.to_csv("firms_california_combined.csv", index=False)
print("Saved!")
