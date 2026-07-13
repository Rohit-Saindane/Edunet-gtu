# Saturday, July 11, 2026
# Linear Regression Model: Predict total carbon footprint (GWP) of IT equipment

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "boavizta-data-us.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing dataset at {csv_path}. Please run download_data.py first.")

df = pd.read_csv(csv_path)

print("Dataset Columns:", df.columns.tolist())
print("\nInitial Null Values:\n", df.isnull().sum())

print("\nDataset generalizations and analysis:")
print("- gwp_total represents the total Greenhouse Warming Potential carbon footprint of the device.")
print("- weight (kg) is directly proportional to the carbon footprint (gwp_total).")
print("- lifetime (years) represents the device lifespan, which relates to how long the device is used.")
print("- category represents the type of hardware (Workplace, Datacenter, Home).")

# Handling missing values
df['lifetime'] = df['lifetime'].fillna(df.groupby('category')['lifetime'].transform('median'))
df['lifetime'] = df['lifetime'].fillna(df['lifetime'].median())

df['weight'] = df['weight'].fillna(df.groupby('category')['weight'].transform('median'))
df['weight'] = df['weight'].fillna(df['weight'].median())

print("\nNull Values after handling:\n", df.isnull().sum())

# Feature Engineering
df['weight_lifetime_ratio'] = df['weight'] / df['lifetime']

# Visualisations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="weight", y="gwp_total", hue="category")
plt.title("Device Weight vs GWP Total")
plt.savefig(os.path.join(script_dir, "boavizta_weight_vs_gwp.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="category", y="gwp_total", hue="category", legend=False)
plt.title("GWP Total by Category")
plt.yscale("log")
plt.savefig(os.path.join(script_dir, "boavizta_gwp_by_category.png"))
plt.close()

# Encoding and Scaling
df_encoded = pd.get_dummies(df, columns=['category', 'manufacturer'], drop_first=True)

X = df_encoded.drop(columns=['gwp_total'])
y_log = np.log1p(df_encoded['gwp_total'])

bool_cols = X.select_dtypes(include=['bool']).columns
X[bool_cols] = X[bool_cols].astype(int)

num_features = ['weight', 'lifetime', 'weight_lifetime_ratio']
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Model Training
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_train_pred_log = lr.predict(X_train)
y_test_pred_log = lr.predict(X_test)

y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

log_test_r2 = r2_score(y_test, y_test_pred_log)
log_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_log))

orig_test_r2 = r2_score(y_test_orig, y_test_pred)
orig_test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))

print("\nLinear Regression Evaluation Metrics:")
print(f"Log-transformed Test R2: {log_test_r2:.4f}")
print(f"Log-transformed Test RMSE: {log_test_rmse:.4f}")
print(f"Original Space Test R2: {orig_test_r2:.4f}")
print(f"Original Space Test RMSE: {orig_test_rmse:.2f} kg CO2 eq")

# Coefficients Analysis
print("\nIntercept (Base Log GWP):", round(lr.intercept_, 2))
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\nModel Coefficients:")
print(coef_df)

print("\nInsights and Explanations:")
print("1. Why we did what we did:")
print("   - Null Handling: Imputed missing values based on category medians because device type determines physical characteristics.")
print("   - Feature Engineering: Engineered weight_lifetime_ratio to represent the mass of material used per year of life.")
print("   - Log-transformation: Applied log transformation on GWP to address massive variance between laptops (100 kg GWP) and servers (10,000 kg GWP).")
print("   - Encoding: Used pd.get_dummies to convert category and manufacturer columns into numeric format.")
print("   - Scaling: Applied StandardScaler to keep numeric variables on the same scale.")
print("2. Model performance and errors:")
print(f"   - The R2 score in the original space is {orig_test_r2:.4f}, meaning the model explains about {orig_test_r2*100:.1f}% of GWP variance.")
print(f"   - The RMSE of {orig_test_rmse:.2f} shows our model predictions deviate by about {orig_test_rmse:.1f} kg CO2 on average.")
