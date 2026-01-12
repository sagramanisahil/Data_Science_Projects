import numpy as np
import pandas as pd

df = pd.read_csv("TESLA.csv")
print("Shape: ", df.shape)
print(df.head())
print(df.info)

print(df.isnull().sum())
print(df.duplicated().sum())
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)
df = df.sort_values(by='Date')
df.drop(columns=['Adj Close'], inplace=True)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].round(2)

df.to_csv("Cleaned Data.csv")