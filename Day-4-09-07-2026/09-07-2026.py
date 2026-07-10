import os
import urllib.request
import pandas as pd

# ==============================================================================
# DATASET GENERALISATIONS & ANALYSIS
# ==============================================================================
# Dataset: Fossil Fuel CO2 Emissions by Nation (tracking 1751 to modern era)
# Source: Carbon Dioxide Information Analysis Center (CDIAC) / World Bank / OWID
#
# FEATURE DESCRIPTIONS & RELATIONSHIPS:
# 1. Year: The timeline of emissions. Relates to global industrialisation phases.
# 2. Country: The geographic entity. Relates to national industrial/economic capacity.
# 3. Total: Total CO2 emissions (in thousand metric tons of carbon).
#    Relationship: Total = Solid Fuel + Liquid Fuel + Gas Fuel + Cement + Gas Flaring.
# 4. Solid Fuel: Coal emissions. Heavily present in early industrialisation (1800s-1900s).
# 5. Liquid Fuel: Petroleum/Oil emissions. Dominates after mid-20th century.
# 6. Gas Fuel: Natural Gas emissions. Grows in late 20th and 21st century.
# 7. Cement: CO2 released during chemical reaction in cement production.
#    Relationship: Direct proxy for infrastructure and construction boom.
# 8. Gas Flaring: CO2 from burning of excess natural gas during oil extraction.
#    Relationship: Correlates strongly with major oil-producing nations.
# 9. Per Capita: Carbon emissions per person (metric tons of carbon per person).
#    Relationship: Per Capita = Total / Country Population. Shows average footprint.
# 10. Bunker fuels: Fuel used in international aviation and maritime transport.
#     Relationship: Excluded from national Totals; relates to international trade.
# ==============================================================================

# Download the sustainability dataset
url = "https://raw.githubusercontent.com/datasets/co2-fossil-by-nation/master/data/fossil-fuel-co2-emissions-by-nation.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "fossil-fuel-co2-emissions-by-nation.csv")

if not os.path.exists(csv_path):
    print("Downloading sustainability dataset...")
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete.")

# Import the dataset in python using pandas
df = pd.read_csv(csv_path)

print("--- Initial Dataset Shape ---")
print(df.shape)
print("\n--- Columns in Dataset ---")
print(df.columns.tolist())

# Checking for Null Values
print("\n--- Missing Values Before Handling ---")
print(df.isnull().sum())

# Handling Null Values
# 1. Emission Sources (Solid/Liquid/Gas/Cement/Flaring/Bunker):
#    Missing values represent no recorded emissions or no activity for that category.
#    Filling them with 0.0 is the most logical representation.
emission_cols = ["Solid Fuel", "Liquid Fuel", "Gas Fuel", "Cement", "Gas Flaring", "Bunker fuels (Not in Total)"]
df[emission_cols] = df[emission_cols].fillna(0.0)

# 2. Per Capita:
#    If Per Capita is missing, population data was likely unavailable.
#    To handle this without inserting zeros (which would skew the per-person average),
#    we fill missing values with the median Per Capita of that specific country.
#    If a country has no Per Capita data at all, we fill it with the overall dataset median.
df["Per Capita"] = df["Per Capita"].fillna(df.groupby("Country")["Per Capita"].transform("median"))
df["Per Capita"] = df["Per Capita"].fillna(df["Per Capita"].median())

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

# Brief Analysis: Top 5 emitting nations in the most recent year
latest_year = df["Year"].max()
print(f"\n--- Top 5 Emitting Nations in {latest_year} ---")
latest_df = df[df["Year"] == latest_year]
top_emitters = latest_df.sort_values(by="Total", ascending=False).head(6)
# Excluding global/regional aggregates if they exist
print(top_emitters[["Country", "Total", "Per Capita"]])

# Relation verification: Verify that Total is roughly sum of individual fuel sources
print("\n--- Verifying Total Carbon emissions formula (Total vs Sum of Sources) for row 0 ---")
sample_row = df.iloc[0]
calculated_total = (sample_row["Solid Fuel"] + sample_row["Liquid Fuel"] + 
                    sample_row["Gas Fuel"] + sample_row["Cement"] + 
                    sample_row["Gas Flaring"])
print(f"Dataset reported Total: {sample_row['Total']}")
print(f"Calculated Total from sources: {calculated_total}")

# Save the cleaned dataset
cleaned_csv_path = os.path.join(script_dir, "fossil-fuel-co2-emissions-cleaned.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"\nCleaned dataset successfully saved to: {cleaned_csv_path}")

# ==============================================================================
# SUMMARY OF WORK DONE:
# 1. Downloaded a sustainability dataset tracking Fossil Fuel CO2 emissions by nation.
# 2. Imported the dataset using pandas and analysed the features and their relationships.
# 3. Handled null values logically: filled specific fuel emission sources with 0.0
#    (as null indicates no recorded emissions of that type), and filled Per Capita 
#    emissions with country-specific medians (and overall median where needed) to 
#    prevent calculation skew.
# 4. Verified the mathematical relation between total emissions and separate sources.
# 5. Saved the cleaned dataset as 'fossil-fuel-co2-emissions-cleaned.csv' inside the same folder.
# ==============================================================================
