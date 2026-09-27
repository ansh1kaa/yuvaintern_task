# ============================================================
# WEEK 3 - EXPLORATORY DATA ANALYSIS ON LOGISTICS PERFORMANCE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
file_path = "Delivery_Logistics_Cleaned.csv"

df = pd.read_csv("Delivery_Logistics_Cleaned.csv")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 2. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()


# ------------------------------------------------------------
# 3. COLUMN NAMES
# ------------------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print("\nTotal Missing Values:", missing.sum())


# ------------------------------------------------------------
# 5. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# ------------------------------------------------------------
# 6. NUMERICAL SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns

summary = df[numeric_columns].describe().T

summary["median"] = df[numeric_columns].median()

summary["IQR"] = (
    df[numeric_columns].quantile(0.75)
    - df[numeric_columns].quantile(0.25)
)

summary = summary[
    ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "IQR"]
]

print(summary.round(2))


# ------------------------------------------------------------
# 7. IMPORTANT LOGISTICS METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("KEY LOGISTICS METRICS")
print("=" * 70)

for col in [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost",
    "delivery_rating"
]:
    
    if col in df.columns:
        print(
            f"{col}: "
            f"Mean = {df[col].mean():.2f}, "
            f"Median = {df[col].median():.2f}, "
            f"Min = {df[col].min():.2f}, "
            f"Max = {df[col].max():.2f}"
        )


# ------------------------------------------------------------
# 8. DELAY ANALYSIS
# ------------------------------------------------------------

if "delayed" in df.columns:

    print("\n" + "=" * 70)
    print("DELAY ANALYSIS")
    print("=" * 70)

    print(df["delayed"].value_counts())

    print("\nDelay Percentage:")

    delay_percentage = (
        df["delayed"].value_counts(normalize=True) * 100
    )

    print(delay_percentage.round(2))


# ------------------------------------------------------------
# 9. DELIVERY STATUS DISTRIBUTION
# ------------------------------------------------------------

if "delivery_status" in df.columns:

    print("\n" + "=" * 70)
    print("DELIVERY STATUS DISTRIBUTION")
    print("=" * 70)

    status_counts = df["delivery_status"].value_counts()

    print(status_counts)

    print("\nPercentage:")
    print(
        (df["delivery_status"].value_counts(normalize=True) * 100)
        .round(2)
    )


# ------------------------------------------------------------
# 10. HISTOGRAM - DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    df["delivery_time_hours"],
    bins=30,
    kde=True
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Hours)")
plt.ylabel("Number of Deliveries")
plt.tight_layout()

plt.savefig("01_delivery_time_distribution.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 11. HISTOGRAM - DELIVERY COST
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    df["delivery_cost"],
    bins=30,
    kde=True
)

plt.title("Distribution of Delivery Cost")
plt.xlabel("Delivery Cost")
plt.ylabel("Number of Deliveries")
plt.tight_layout()

plt.savefig("02_delivery_cost_distribution.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 12. HISTOGRAM - DISTANCE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    df["distance_km"],
    bins=30,
    kde=True
)

plt.title("Distribution of Delivery Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Number of Deliveries")
plt.tight_layout()

plt.savefig("03_distance_distribution.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 13. BOX PLOT - DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

sns.boxplot(
    x=df["delivery_time_hours"]
)

plt.title("Box Plot of Delivery Time")
plt.xlabel("Delivery Time (Hours)")
plt.tight_layout()

plt.savefig("04_delivery_time_boxplot.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 14. BOX PLOT - DELIVERY COST
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

sns.boxplot(
    x=df["delivery_cost"]
)

plt.title("Box Plot of Delivery Cost")
plt.xlabel("Delivery Cost")
plt.tight_layout()

plt.savefig("05_delivery_cost_boxplot.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 15. BOX PLOT - DISTANCE
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

sns.boxplot(
    x=df["distance_km"]
)

plt.title("Box Plot of Delivery Distance")
plt.xlabel("Distance (km)")
plt.tight_layout()

plt.savefig("06_distance_boxplot.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 16. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS USING IQR")
print("=" * 70)


def detect_outliers(column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit)
        | (df[column] > upper_limit)
    ]

    return Q1, Q3, IQR, lower_limit, upper_limit, len(outliers)


outlier_results = []

for column in [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "delivery_cost"
]:

    if column in df.columns:

        Q1, Q3, IQR, lower, upper, count = detect_outliers(column)

        outlier_results.append([
            column,
            Q1,
            Q3,
            IQR,
            lower,
            upper,
            count
        ])

outlier_table = pd.DataFrame(
    outlier_results,
    columns=[
        "Variable",
        "Q1",
        "Q3",
        "IQR",
        "Lower Bound",
        "Upper Bound",
        "Outlier Count"
    ]
)

print(outlier_table.round(2))


# ------------------------------------------------------------
# 17. DISTANCE VS DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_time_hours",
    alpha=0.4
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Hours)")
plt.tight_layout()

plt.savefig("07_distance_vs_delivery_time.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 18. DISTANCE VS DELIVERY COST
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_cost",
    alpha=0.4
)

plt.title("Distance vs Delivery Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Cost")
plt.tight_layout()

plt.savefig("08_distance_vs_delivery_cost.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 19. DELIVERY TIME VS EXPECTED TIME
# ------------------------------------------------------------

if "expected_time_hours" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x="expected_time_hours",
        y="delivery_time_hours",
        alpha=0.4
    )

    plt.title("Actual Delivery Time vs Expected Delivery Time")
    plt.xlabel("Expected Delivery Time (Hours)")
    plt.ylabel("Actual Delivery Time (Hours)")
    plt.tight_layout()

    plt.savefig("09_actual_vs_expected_time.png", dpi=300)

    plt.show()


# ------------------------------------------------------------
# 20. AVERAGE DELIVERY TIME BY REGION
# ------------------------------------------------------------

if "region" in df.columns:

    region_time = (
        df.groupby("region")["delivery_time_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    region_time.plot(kind="bar")

    plt.title("Average Delivery Time by Region")
    plt.xlabel("Region")
    plt.ylabel("Average Delivery Time (Hours)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("10_avg_delivery_time_region.png", dpi=300)

    plt.show()

    print("\nAverage Delivery Time by Region:")
    print(region_time.round(2))


# ------------------------------------------------------------
# 21. AVERAGE COST BY DELIVERY MODE
# ------------------------------------------------------------

if "delivery_mode" in df.columns:

    mode_cost = (
        df.groupby("delivery_mode")["delivery_cost"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    mode_cost.plot(kind="bar")

    plt.title("Average Delivery Cost by Delivery Mode")
    plt.xlabel("Delivery Mode")
    plt.ylabel("Average Delivery Cost")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("11_avg_cost_delivery_mode.png", dpi=300)

    plt.show()

    print("\nAverage Cost by Delivery Mode:")
    print(mode_cost.round(2))


# ------------------------------------------------------------
# 22. AVERAGE RATING BY DELIVERY PARTNER
# ------------------------------------------------------------

if "delivery_partner" in df.columns:

    partner_rating = (
        df.groupby("delivery_partner")["delivery_rating"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))

    partner_rating.plot(kind="bar")

    plt.title("Average Delivery Rating by Delivery Partner")
    plt.xlabel("Delivery Partner")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("12_avg_rating_partner.png", dpi=300)

    plt.show()


# ------------------------------------------------------------
# 23. DELAY BY VEHICLE TYPE
# ------------------------------------------------------------

if "vehicle_type" in df.columns and "delayed" in df.columns:

    vehicle_delay = (
        df.groupby("vehicle_type")["delayed"]
        .count()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    vehicle_delay.plot(kind="bar")

    plt.title("Delivery Delay Records by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("13_delay_vehicle_type.png", dpi=300)

    plt.show()


# ------------------------------------------------------------
# 24. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

correlation_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"
]

available_columns = [
    col for col in correlation_columns
    if col in df.columns
]

corr = df[available_columns].corr()

print(corr.round(2))


# ------------------------------------------------------------
# 25. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    fmt=".2f"
)

plt.title("Correlation Matrix of Logistics Performance Metrics")
plt.tight_layout()

plt.savefig("14_correlation_heatmap.png", dpi=300)

plt.show()
# ------------------------------------------------------------
# 26. DELAY RATE BY REGION
# ------------------------------------------------------------

if "region" in df.columns and "delayed" in df.columns:

    delay_region = (
        df.groupby("region")["delayed"]
        .apply(
            lambda x: x.astype(str)
            .str.lower()
            .isin(["yes", "true", "1", "delayed"])
            .mean() * 100
        )
        .sort_values(ascending=False)
    )

    print("\nDelay Rate by Region (%):")
    print(delay_region.round(2))

    plt.figure(figsize=(9, 5))

    delay_region.plot(kind="bar")

    plt.title("Delay Rate by Region")
    plt.xlabel("Region")
    plt.ylabel("Delay Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("15_delay_rate_region.png", dpi=300)

    plt.show()

# ------------------------------------------------------------
# 27. FINAL DATASET SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL EDA SUMMARY")
print("=" * 70)

print("Number of records:", len(df))
print("Number of columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

if "delivery_time_hours" in df.columns:
    print(
        "Average delivery time:",
        round(df["delivery_time_hours"].mean(), 2),
        "hours"
    )

if "delivery_cost" in df.columns:
    print(
        "Average delivery cost:",
        round(df["delivery_cost"].mean(), 2)
    )

if "delivery_rating" in df.columns:
    print(
        "Average delivery rating:",
        round(df["delivery_rating"].mean(), 2)
    )

print("\nEDA COMPLETE!")
print("All charts have been saved as PNG files in the notebook folder.")