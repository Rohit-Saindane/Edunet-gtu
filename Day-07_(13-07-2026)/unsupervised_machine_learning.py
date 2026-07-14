# Monday, July 13, 2026
# K-Means Clustering on Global Sustainable Energy

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "global-data-on-sustainable-energy.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing dataset at {csv_path}")

df = pd.read_csv(csv_path)

# Cleaning
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

features = [
    'Access_to_Electricity_Pct',
    'Renewable_Energy_Share_Pct',
    'CO2_Emissions_Mton',
    'GDP_Per_Capita_USD',
    'Renewable_Energy_Capacity_Per_Capita'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# GDP vs Renewables
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='GDP_Per_Capita_USD',
    y='Renewable_Energy_Share_Pct',
    hue='Access_to_Electricity_Pct',
    palette='viridis',
    size='CO2_Emissions_Mton',
    sizes=(40, 400),
    alpha=0.8
)
plt.title("GDP vs. Renewable Energy Share")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "gdp_vs_renewables.png"), dpi=300)
plt.close()

# Elbow Curve
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', color='teal')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "elbow_method.png"), dpi=300)
plt.close()

# K-Means
kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans_model.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {silhouette:.4f}")

# Plot Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='GDP_Per_Capita_USD',
    y='Renewable_Energy_Share_Pct',
    hue='Cluster',
    palette='Set1',
    s=100,
    edgecolor='black'
)

for idx, row in df.iterrows():
    plt.text(
        row['GDP_Per_Capita_USD'] + 1000,
        row['Renewable_Energy_Share_Pct'] - 1,
        row['Country'],
        fontsize=8
    )

plt.title("Clusters (K=3)")
plt.xlabel("GDP Per Capita (USD)")
plt.ylabel("Renewable Energy Share (%)")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "kmeans_clusters.png"), dpi=300)
plt.close()

print("\n--- Cluster Summary (Mean Unscaled Values) ---")
print(df.groupby('Cluster')[features].mean())

print("\n--- Country Groups ---")
for c_id in range(3):
    countries = df[df['Cluster'] == c_id]['Country'].tolist()
    print(f"Cluster {c_id}: {', '.join(countries)}")
