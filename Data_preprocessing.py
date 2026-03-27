import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("C:\\Users\\Crexed-41\\Downloads\\AI_jobs.csv")


# explore datasets
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())
print("\nStatistics:\n", df.describe())


# handling missing values
print("\nMissing Values:\n", df.isnull().sum())
print(df.dropna())

# fill numericall and categorical coolumns with mean and mode
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\nMissing values handled!")


# remove duplicates
print("\nDuplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())


# encodes categorical data
label_encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical data encoded!")


# feature scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("\nScaled Data Sample:\n", df_scaled.head())


# histogram
df.hist(figsize=(12, 12), bins=20, color="lightgreen", edgecolor="green")
plt.suptitle("Jobs Data Distribution", fontsize=16)
plt.tight_layout()
plt.show()


# SCATTER PLOT
plt.figure(figsize=(6, 5))
plt.scatter(df["years_experience"], df["salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary Relationship")
plt.tight_layout()
plt.show()


# SCATTER MATRIX
numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12, 12))
scatter_matrix(numeric_df, figsize=(12, 12))
plt.suptitle("Scatter Matrix of All Features")
plt.show()
