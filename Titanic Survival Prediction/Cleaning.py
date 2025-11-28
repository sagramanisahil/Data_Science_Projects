import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
print("Shape: ", df.shape)
print(df.isnull().sum())
print(df.head())
df.drop_duplicates(inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace = True)

print(df.isnull().sum())

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

df[['Name', 'Title', 'FamilySize', 'IsAlone']].head(10)

# Encoding Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

df = pd.get_dummies(df, columns=['Title'], drop_first=True)
print(df.head())
print(df.info)

df = df.to_csv("Cleaned_Data.csv", index=False)

