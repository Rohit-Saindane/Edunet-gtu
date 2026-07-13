import os
import urllib.request

script_dir = os.path.dirname(os.path.abspath(__file__))

boavizta_url = "https://raw.githubusercontent.com/Boavizta/environmental-footprint-data/main/boavizta-data-us.csv"
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

boavizta_path = os.path.join(script_dir, "boavizta-data-us.csv")
co2_path = os.path.join(script_dir, "owid-co2-data.csv")

print("Downloading Boavizta IT Environmental Footprint dataset...")
try:
    req = urllib.request.Request(boavizta_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(boavizta_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Successfully downloaded and saved to: {boavizta_path}")
except Exception as e:
    print(f"Error downloading Boavizta dataset: {e}")

print("Downloading OWID Global CO2 Emissions dataset...")
try:
    req = urllib.request.Request(co2_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(co2_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Successfully downloaded and saved to: {co2_path}")
except Exception as e:
    print(f"Error downloading CO2 dataset: {e}")
