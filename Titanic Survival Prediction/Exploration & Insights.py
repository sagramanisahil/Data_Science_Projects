# ==========================
# IMPORT LIBRARIES
# ==========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("Cleaned_Data.csv")

print("Columns:", df.columns)
print("FIRST 5 ROWS:\n", df.head())
print("\nLAST 5 ROWS:\n", df.tail())
print("\nSAMPLE 10 ROWS:\n", df.sample(10))
print("\nMISSING VALUES:\n", df.isnull().sum())
# ==========================
# BASIC STATISTICS
# ==========================
print("\nTOTAL SURVIVED / DIED:\n", df['Survived'].value_counts())
print("\nAVERAGE AGE:", round(df['Age'].mean(),2))
print("\nSURVIVAL RATE BY GENDER:\n", df.groupby('Sex')['Survived'].mean())
print("\nSURVIVAL RATE BY CLASS:\n", df.groupby('Pclass')['Survived'].mean())
# ==========================
# FEATURE ENGINEERING
# ==========================
# Age Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,19,60,80], labels=['Child','Teen','Adult','Senior'])
# Surname
df['Surname'] = df['Name'].str.split(',').str[0]
# ==========================
# 1. Survival by Age Groups
# ==========================
age_group_survival = df.groupby('AgeGroup')['Survived'].value_counts().unstack()
age_group_survival.plot(kind='bar', stacked=True, color=['red','green'])
plt.title("Survival by Age Groups")
plt.ylabel("Count")
plt.show()
# ==========================
# 2. Age Distribution (Histogram + KDE)
# ==========================
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title("Age Boxplot")
plt.show()
# ==========================
# 3. Age Distribution by Survival
# ==========================
sns.histplot(df[df['Survived']==0]['Age'], bins=30, color='red', alpha=0.6, label='Died')
sns.histplot(df[df['Survived']==1]['Age'], bins=30, color='green', alpha=0.6, label='Survived')
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()
sns.boxplot(x='Survived', y='Age', data=df, palette=['red','green'])
plt.title("Boxplot: Age vs Survival")
plt.show()
# ==========================
# 4. Survival by Embarked Port
# ==========================
embarked_survival = df.groupby('Embarked')['Survived'].value_counts().unstack()
embarked_survival.plot(kind='bar', stacked=True, color=['red','green'])
plt.title("Survival by Embarked Port")
plt.ylabel("Count")
plt.show()
# ==========================
# 5. Pclass vs Gender vs Survival
# ==========================
pclass_gender_survival = df.groupby(['Pclass','Sex'])['Survived'].value_counts().unstack()
pclass_gender_survival.plot(kind='bar', stacked=True)
plt.title("Pclass vs Gender vs Survival")
plt.ylabel("Count")
plt.show()
# ==========================
# 6. Fare Distribution
# ==========================
sns.histplot(df['Fare'], bins=30, kde=True, color='orange')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()
sns.boxplot(x=df['Fare'], color='lightcoral')
plt.title("Fare Boxplot")
plt.show()

# ==========================
# 7. Fare vs Survival
# ==========================
sns.boxplot(x='Survived', y='Fare', data=df, palette=['red','green'])
plt.title("Fare vs Survival")
plt.ylabel("Fare")
plt.show()
# ==========================
# 8. Survival by Titles (One-hot)
# ==========================
title_columns = ['Title_Mr','Title_Miss','Title_Mrs','Title_Rare']
title_survival = df[title_columns + ['Survived']].groupby(title_columns).sum()
title_survival.plot(kind='bar', stacked=True)
plt.title("Survival by Title")
plt.ylabel("Count")
plt.show()
# ==========================
# 9. Family Size Impact
# ==========================
family_survival = df.groupby('FamilySize')['Survived'].mean()
plt.plot(family_survival.index, family_survival.values, marker='o', color='purple')
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.show()
# ==========================
# 10. Being Alone Impact
# ==========================
alone_survival = df.groupby('IsAlone')['Survived'].mean()
plt.bar(['With Family','Alone'], alone_survival.values, color=['skyblue','salmon'])
plt.title("Impact of Being Alone on Survival")
plt.ylabel("Survival Rate")
plt.show()
# ==========================
# 11. Correlation Heatmap
# ==========================
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
# ==========================
# 12. Age vs Fare colored by Survival
# ==========================
colors = df['Survived'].map({0:'red',1:'green'})
plt.scatter(df['Age'], df['Fare'], c=colors, alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare colored by Survival")
plt.show()
# ==========================
# 13. Survival Probability by Age (Lineplot)
# ==========================
age_bins = pd.cut(df['Age'], bins=range(0,81,10))
age_survival = df.groupby(age_bins)['Survived'].mean()
plt.plot(range(len(age_survival)), age_survival.values, marker='o', color='blue')
plt.xticks(range(len(age_survival)), [f"{int(interval.left)}-{int(interval.right)}" for interval in age_survival.index], rotation=45)
plt.xlabel("Age Group")
plt.ylabel("Survival Probability")
plt.title("Survival Probability by Age")
plt.show()
# ==========================
# 14. Fare Across Embarked Ports
# ==========================
ports = df['Embarked'].dropna().unique()
fare_data = [df[df['Embarked']==p]['Fare'] for p in ports]
plt.boxplot(fare_data, labels=ports)
plt.title("Fare Distribution Across Embarked Ports")
plt.ylabel("Fare")
plt.show()
# ==========================
# 15. Most Common Surnames
# ==========================
top_survivors = df[df['Survived']==1]['Surname'].value_counts().head(10)
top_non_survivors = df[df['Survived']==0]['Surname'].value_counts().head(10)
plt.barh(top_survivors.index[::-1], top_survivors.values[::-1], color='green')
plt.title("Top 10 Surnames of Survivors")
plt.show()
plt.barh(top_non_survivors.index[::-1], top_non_survivors.values[::-1], color='red')
plt.title("Top 10 Surnames of Non-Survivors")
plt.show()
# ==========================
# 16. Pie Chart of Embarked
# ==========================
df['Embarked'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue','lightgreen','salmon'])
plt.title("Passenger Distribution by Embarked Port")
plt.ylabel('')
plt.show()
# ==========================
# 18. Pairplot of Key Features
# ==========================
sns.pairplot(df[['Age','Fare','Pclass','Survived']], hue='Survived', palette={0:'red',1:'green'})
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()
# ==========================
# 19. Violin Plot: Age vs Pclass
# ==========================
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=df, split=True, palette={0:'red',1:'green'})
plt.title("Age Distribution by Pclass and Survival")
plt.show()
# ==========================
# 20. KDE Plots by Gender
# ==========================
sns.kdeplot(df[df['Sex']==0]['Age'], label='Male', shade=True, color='blue')
sns.kdeplot(df[df['Sex']==1]['Age'], label='Female', shade=True, color='pink')
plt.title("Age Distribution by Gender")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show()
# ==========================
# 21. Stacked Bar: Survival by Titles (percent)
# ==========================
title_survived_pct = df[title_columns + ['Survived']].groupby(title_columns).sum().T
title_survived_pct.plot(kind='bar', stacked=True, colormap='Set2')
plt.title("Stacked Bar: Survival by Titles")
plt.ylabel("Count")
plt.show()
# ==========================
# 22. Line Plot: Survival Rate by AgeGroup
# ==========================
agegroup_surv_rate = df.groupby('AgeGroup')['Survived'].mean()
plt.plot(agegroup_surv_rate.index, agegroup_surv_rate.values, marker='o', color='purple')
plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.show()
# ==========================
# 23. Boxplot: Fare by Pclass
# ==========================
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set3')
plt.title("Fare Distribution by Pclass")
plt.show()
# ==========================
# 24. Histogram: Family Size
# ==========================
sns.histplot(df['FamilySize'], bins=15, kde=False, color='lightblue')
plt.title("Family Size Distribution")
plt.xlabel("Family Size")
plt.ylabel("Count")
plt.show()
# ==========================
# 25. Bar Plot: IsAlone vs Survival Rate
# ==========================
sns.barplot(x='IsAlone', y='Survived', data=df, palette=['skyblue','salmon'])
plt.xticks([0,1], ['With Family','Alone'])
plt.title("IsAlone vs Survival Rate")
plt.ylabel("Survival Rate")
plt.show()
# ==========================
# 26. Scatter Plot: Age vs Fare, Pclass Hue
# ==========================
sns.scatterplot(x='Age', y='Fare', hue='Pclass', style='Survived', data=df, palette='Set1', alpha=0.6, s=80)
plt.title("Scatter Plot: Age vs Fare vs Pclass & Survival")
plt.show()
