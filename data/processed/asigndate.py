import pandas as pd
import numpy as np

# Load your file
df = pd.read_csv("test.csv")

# Create daily date range like paper period
date_pool = pd.date_range(start="2002-01-01", end="2017-12-31", freq="D")

print("Rows in dataset:", len(df))
print("Available dates:", len(date_pool))

# Repeat dates until all rows get a date
repeated_dates = np.resize(date_pool, len(df))

# Assign to dataframe
df["Date"] = repeated_dates

# Optional: sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Save
df.to_csv("dataset_with_dates.csv", index=False)

print(df)