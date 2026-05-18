import pandas as pd

df = pd.read_csv("data/processed/cleaned.csv")

print(df.head())

print("\nDistribution:\n", df['sentiment'].value_counts())