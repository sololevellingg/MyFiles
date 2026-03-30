import pandas as pd
import numpy as np

np.random.seed(42)

# Date range: Jan 2021 - Dec 2023 (3 years of daily data)
dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")

categories = ["Electronics", "Clothing", "Grocery", "Furniture", "Sports"]
regions    = ["North", "South", "East", "West"]

rows = []
for date in dates:
    for category in categories:
        for region in regions:
            # Base sales per category
            base = {"Electronics": 12000, "Clothing": 8000,
                    "Grocery": 15000, "Furniture": 5000, "Sports": 6000}[category]

            # Weekend boost
            weekend = 1.25 if date.weekday() >= 5 else 1.0

            # Festival seasons boost (Oct-Dec)
            festival = 1.40 if date.month in [10, 11, 12] else 1.0

            # Summer boost for Sports & Clothing (Apr-Jun)
            summer = 1.20 if (date.month in [4, 5, 6] and
                              category in ["Sports", "Clothing"]) else 1.0

            # Year-on-year growth (~8% per year)
            growth = 1 + 0.08 * (date.year - 2021)

            # Regional multiplier
            region_mult = {"North": 1.1, "South": 1.0,
                           "East": 0.9, "West": 1.05}[region]

            # Random noise
            noise = np.random.normal(1.0, 0.08)

            sales    = int(base * weekend * festival * summer * growth * region_mult * noise)
            units    = int(sales / np.random.uniform(200, 600))
            discount = round(np.random.uniform(0, 0.25), 2)
            profit   = round(sales * np.random.uniform(0.10, 0.30), 2)

            rows.append({
                "date":      date.strftime("%Y-%m-%d"),
                "category":  category,
                "region":    region,
                "sales":     max(sales, 0),
                "units_sold": max(units, 0),
                "discount":  discount,
                "profit":    profit
            })

df = pd.DataFrame(rows)
df.to_csv("retail_sales_data.csv", index=False)
print(f"Dataset created: {len(df):,} rows")
print(df.head())
print(df.describe())
