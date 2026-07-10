import os
import urllib.request
import pandas as pd

url = "https://raw.githubusercontent.com/datasets/co2-fossil-by-nation/master/data/fossil-fuel-co2-emissions-by-nation.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "fossil-fuel-co2-emissions-by-nation.csv")

if not os.path.exists(csv_path):
    urllib.request.urlretrieve(url, csv_path)

df = pd.read_csv(csv_path)

print("Dataset Columns:", df.columns.tolist())
print("Initial Null Values:\n", df.isnull().sum())

print("\nDataset analysis and feature relations:")
print("- Total is the sum of Solid Fuel, Liquid Fuel, Gas Fuel, Cement, and Gas Flaring emissions.")
print("- Year tracks emissions over time, showing industrial growth.")
print("- Per Capita represents Total emissions divided by country population.")

emission_cols = ["Solid Fuel", "Liquid Fuel", "Gas Fuel", "Cement", "Gas Flaring", "Bunker fuels (Not in Total)"]
df[emission_cols] = df[emission_cols].fillna(0.0)

df["Per Capita"] = df["Per Capita"].fillna(df.groupby("Country")["Per Capita"].transform("median"))
df["Per Capita"] = df["Per Capita"].fillna(df["Per Capita"].median())

print("\nNull Values after handling:\n", df.isnull().sum())

latest_year = df["Year"].max()
print(f"\nTop 5 Emitting Nations in {latest_year}:")
print(df[df["Year"] == latest_year].sort_values(by="Total", ascending=False).head(5))

print("\nSummary:")
print("- Loaded carbon emissions dataset and handled missing values.")
print("- Filled null emission columns with 0.0.")
print("- Filled missing Per Capita values with median country values.")
print("- Exported clean data.")

df.to_csv(os.path.join(script_dir, "fossil-fuel-co2-emissions-cleaned.csv"), index=False)
