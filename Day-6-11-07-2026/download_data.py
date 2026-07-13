import os
import urllib.request
import pandas as pd
import io

script_dir = os.path.dirname(os.path.abspath(__file__))

boavizta_url = "https://raw.githubusercontent.com/Boavizta/environmental-footprint-data/main/boavizta-data-us.csv"
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

boavizta_path = os.path.join(script_dir, "boavizta-data-us.csv")
co2_path = os.path.join(script_dir, "owid-co2-data.csv")

print("Downloading and processing Boavizta IT Environmental Footprint dataset...")
try:
    req = urllib.request.Request(boavizta_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
    # Keep only 5 columns (well between 5 and 15)
    cols = ['manufacturer', 'category', 'gwp_total', 'lifetime', 'weight']
    df_small = df[cols]
    df_small.to_csv(boavizta_path, index=False)
    print(f"Successfully processed and saved to: {boavizta_path} (Columns: {df_small.shape[1]}, Rows: {df_small.shape[0]})")
except Exception as e:
    print(f"Error: {e}")

print("Downloading and processing OWID Global CO2 Emissions dataset...")
try:
    req = urllib.request.Request(co2_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
    # Filter for years 2000-2022, countries only (where iso_code is not null)
    df_filtered = df[(df['year'] >= 2000) & (df['year'] <= 2022) & (df['iso_code'].notnull())].copy()
    # Keep only 6 columns
    cols = ['country', 'year', 'population', 'gdp', 'co2_per_capita', 'energy_per_capita']
    df_small = df_filtered[cols]
    df_small.to_csv(co2_path, index=False)
    print(f"Successfully processed and saved to: {co2_path} (Columns: {df_small.shape[1]}, Rows: {df_small.shape[0]})")
except Exception as e:
    print(f"Error: {e}")
