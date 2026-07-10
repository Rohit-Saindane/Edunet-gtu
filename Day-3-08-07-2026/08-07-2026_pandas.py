import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "titanic.csv")

df = pd.read_csv(file_path)

print("--- DataFrame Head ---")
print(df.head())

print("\n--- DataFrame Tail ---")
print(df.tail())

print("\n--- DataFrame Info ---")
df.info()

print("\n--- DataFrame Describe ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
print("\n--- Missing values in Age after fillna ---")
print(df["Age"].isnull().sum())

print("\n--- Groupby Class Mean Fare ---")
print(df.groupby("Pclass")["Fare"].mean())

print("\n--- Sort by Fare ---")
print(df.sort_values(by="Fare", ascending=False).head())

print("\n--- Rename Sex to Gender ---")
df = df.rename(columns={"Sex": "Gender"})
print(df.head(2))

print("\n--- Value Counts for Gender ---")
print(df["Gender"].value_counts())

output_path = os.path.join(script_dir, "titanic_cleaned.csv")
df.to_csv(output_path, index=False)
