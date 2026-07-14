# Day 7: Unsupervised Machine Learning (K-Means Clustering)

## Overview
This task applies the **K-Means Clustering** algorithm to a global sustainable energy dataset to automatically group countries based on their environmental, energy, and economic indicators. It highlights how unsupervised learning can extract patterns and group similar profiles without pre-labeled categories.

---

## Dataset Details
The dataset used is `global-data-on-sustainable-energy.csv`, which has been formatted to a clean, compact version containing indicators for 20 diverse countries:
- **Country**: Name of the country
- **Access_to_Electricity_Pct**: Percentage of the population with access to electricity (%)
- **Renewable_Energy_Share_Pct**: Renewable energy share in total final energy consumption (%)
- **CO2_Emissions_Mton**: Carbon dioxide emissions (Million metric tons)
- **GDP_Per_Capita_USD**: Gross Domestic Product per capita (USD)
- **Renewable_Energy_Capacity_Per_Capita**: Renewable electricity generating capacity per capita (W/capita)

---

## Methodology

### 1. Data Preprocessing
- **Handling Missing Values**: Automatically detects missing values and fills them with the median of the respective columns.
- **Feature Standardization**: Standardizes all features using Scikit-Learn's `StandardScaler` so that features with larger numeric scales (e.g. GDP, CO2) do not dominate the Euclidean distance computations.

### 2. Exploratory Data Analysis (EDA)
- **Correlation Heatmap**: Analyzes relationships between the features (saved as `correlation_heatmap.png`).
- **GDP vs. Renewable Share**: Shows GDP, renewable share, and CO2 footprint distributions (saved as `gdp_vs_renewables.png`).

### 3. Elbow Method
- Plots the Within-Cluster Sum of Squares (WCSS/Inertia) for $K = 1$ to $10$.
- Identifies the "elbow" point to choose the optimal number of clusters (saved as `elbow_method.png`).

### 4. K-Means Clustering
- Fits a K-Means model with the optimal $K = 3$ clusters on the scaled indicators.
- Groups countries and projects them onto a GDP vs. Renewable Share scatter plot (saved as `kmeans_clusters.png`).

### 5. Model Evaluation
- Computes the **Silhouette Score** to measure cluster cohesion and separation.

---

## Results and Interpretation

The algorithm successfully groups countries into 3 distinct clusters:
- **Cluster 0 (Eco-Leaders / High-GDP Renewable Adopters)**: Countries with high GDP per capita, extremely high renewable energy shares, and high renewable capacity (e.g., Norway, Iceland, Sweden).
- **Cluster 1 (Industrialized / High-Emissions Countries)**: Countries with high GDP per capita or high industrial activity accompanied by significant CO2 emissions and lower renewable shares (e.g., United States, China, India, Japan, Germany).
- **Cluster 2 (Developing / Low-Emissions Countries)**: Nations with lower GDP per capita, varying levels of electricity access, but very high renewable energy shares (e.g., Kenya, Ethiopia, Nigeria, Brazil).

### Model Evaluation Metric
- **Silhouette Score**: ~`0.45` to `0.52` (indicates moderately well-defined, meaningful clusters given the dataset variety).
