# Monday, July 13, 2026
# Unsupervised Machine Learning: K-Means Clustering on Global Sustainable Energy Indicators

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set style for seaborn plots
sns.set_theme(style="whitegrid")

# Get path of current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "global-data-on-sustainable-energy.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing dataset at {csv_path}. Please check the files.")

# Load dataset
df = pd.read_csv(csv_path)
print("--- Dataset Loaded Successfully ---")
print(f"Total records: {len(df)}")
print("Columns:", df.columns.tolist())

# Inspect for missing values (if any)
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print("\nMissing values detected. Cleaning data...")
    # Fill numerical columns with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
else:
    print("\nNo missing values found in the dataset.")

# Select numerical features for K-Means Clustering
features = [
    'Access_to_Electricity_Pct',
    'Renewable_Energy_Share_Pct',
    'CO2_Emissions_Mton',
    'GDP_Per_Capita_USD',
    'Renewable_Energy_Capacity_Per_Capita'
]

X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 1. Exploratory Data Analysis (EDA) ---
print("\nGenerating EDA Visualizations...")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Sustainability Indicators", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# Scatter Plot: GDP vs Renewable Energy Share
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
plt.title("GDP vs. Renewable Share (size: CO2 Emissions)", fontsize=13, fontweight='bold')
plt.xlabel("GDP Per Capita (USD)")
plt.ylabel("Renewable Energy Share (%)")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "gdp_vs_renewables.png"), dpi=300)
plt.close()

# --- 2. Determining Optimal Clusters (Elbow Method) ---
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='teal', linewidth=2)
plt.title("Elbow Method to Determine Optimal Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "elbow_method.png"), dpi=300)
plt.close()
print("Elbow curve saved as elbow_method.png.")

# --- 3. Run K-Means Clustering ---
# Based on the elbow curve and dataset size, K=3 is an optimal number of clusters
K_optimal = 3
print(f"\nTraining K-Means Model with K = {K_optimal}...")
kmeans_model = KMeans(n_clusters=K_optimal, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans_model.fit_predict(X_scaled)

# --- 4. Evaluate the Model ---
silhouette = silhouette_score(X_scaled, df['Cluster'])
print(f"Model Evaluation (Silhouette Score): {silhouette:.4f}")

# --- 5. Visualizing Clusters ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='GDP_Per_Capita_USD',
    y='Renewable_Energy_Share_Pct',
    hue='Cluster',
    palette='Set1',
    style='Cluster',
    s=150,
    edgecolor='black',
    alpha=0.9
)

# Label points with Country names
for idx, row in df.iterrows():
    plt.text(
        row['GDP_Per_Capita_USD'] + 1500,
        row['Renewable_Energy_Share_Pct'] - 1.5,
        row['Country'],
        fontsize=8,
        alpha=0.8
    )

plt.title(f"Country Clusters (K-Means K={K_optimal})", fontsize=14, fontweight='bold')
plt.xlabel("GDP Per Capita (USD)")
plt.ylabel("Renewable Energy Share (%)")
plt.legend(title="Cluster ID")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "kmeans_clusters.png"), dpi=300)
plt.close()
print("Clustering scatter plot saved as kmeans_clusters.png.")

# --- 6. Cluster Profiling and Interpretation ---
print("\n--- Cluster Centroid Profiles (Scaled Values) ---")
centroids_scaled = pd.DataFrame(kmeans_model.cluster_centers_, columns=features)
print(centroids_scaled)

print("\n--- Cluster Descriptions (Mean Unscaled Values) ---")
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

print("\n--- Country Groupings by Cluster ---")
for cluster_id in range(K_optimal):
    countries = df[df['Cluster'] == cluster_id]['Country'].tolist()
    print(f"Cluster {cluster_id}: {', '.join(countries)}")

print("\nClustering analysis completed successfully!")
