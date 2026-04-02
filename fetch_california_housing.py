import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


# load datset ---------------

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

print("\n Dataset Information:")
print(f" Shape: {df.shape}")
print(f" Features: {housing.feature_names}")
print(f" Target: Median House Value (in $100,000s)")
print(f" Total samples: {len(df)}")


# Data Exploration

print(df.head())
print(df.describe())
print(df.dtypes)
print(f"\nMissing values:\n{df.isnull().sum()}")
print(df.info())

# visualization ---- distributions

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(df.columns):
    axes[idx].hist(df[col], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[idx].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Frequency")
    axes[idx].axvline(
        df[col].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean : {df[col].mean():.2f}",
    )
    axes[idx].axvline(
        df[col].median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f" Median: {df[col].median():.2f}",
    )
    axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.show()
plt.savefig("California_housung_distributions.png", dpi=100, bbox_inches="tight")


# visualization ----- Correlation Heat-Map

correlation_matrix = df.corr()
corr_with_target = correlation_matrix["MedHouseVal"].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
im = plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, shrink=0.8)

# Add annotations in heat map
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
        )

plt.xticks(
    np.arange(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right",
)
plt.yticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title(
    "Correlation Heatmap - California Housing Features", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("california_housing_correlation.png", dpi=100, bbox_inches="tight")
plt.show()

# visualization scatter plot (features vs target)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, feature in enumerate(housing.feature_names):
    axes[idx].scatter(df[feature], df["MedHouseVal"], alpha=0.3, s=1, c="steelblue")
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel("MedHouseVal", fontsize=10)
    axes[idx].set_title(f"{feature} vs House Value", fontsize=11, fontweight="bold")

    # add trending line
    z = np.polyfit(df[feature], df["MedHouseVal"], 1)
    p = np.poly1d(z)
    axes[idx].plot(
        df[feature].sort_values(),
        p(df[feature].sort_values()),
        "r-",
        linewidth=2,
        label=f'corr: {correlation_matrix.loc[feature, "MedHouseVal"]:.2f}',
    )
    axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.show()
plt.savefig("californai_housing_scatter_plot.png", dpi=100, bbox_inches="tight")

# Data Preprocessing -------

X = df[housing.feature_names]
y = df["MedHouseVal"]

print(f"\n Features shape: {X.shape}")
print(f" Target shape: {y.shape}")

print("\n OUTLIER DETECTION (IQR Method):")
outlier_counts = {}
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].shape[0]
    outlier_counts[col] = outliers
    print(f" {col}: {outliers} outliers ({outliers/len(X)*100:.2f}%)")


# Train-test split ---------

test_sizes = [0.2, 0.25, 0.3]
split_results = []

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    split_results.append(
        {
            "Test Size": f"{test_size*100:.0f}%",
            "Train Size": f"{(1-test_size)*100:.0f}%",
            "Train Samples": len(X_train),
            "Test Samples": len(X_test),
        }
    )
    print(f"\n Test size: {test_size*100:.0f}%")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# using 80/20 split for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Using 80/20 split for modeling")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")
