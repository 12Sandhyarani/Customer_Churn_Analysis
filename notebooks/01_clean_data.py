import pandas as pd
import numpy as np

df = pd.read_csv("../data/Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNulls:\n", df.isnull().sum().sum(), "total nulls")

# TotalCharges is stored as object -> convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nRows with TotalCharges NaN after conversion:", df['TotalCharges'].isnull().sum())
print(df[df['TotalCharges'].isnull()][['customerID', 'tenure', 'TotalCharges']])

# These are all tenure=0 (brand new customers, never billed) -> fill with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop customerID (not predictive), keep for reference separately
customer_ids = df['customerID']
df = df.drop(columns=['customerID'])

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# SeniorCitizen is already 0/1, leave as is

# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Save cleaned data
df.to_csv("../data/cleaned_churn.csv", index=False)
print("\nCleaned data saved. Final shape:", df.shape)
print("\nChurn rate:", df['Churn'].mean().round(3))
