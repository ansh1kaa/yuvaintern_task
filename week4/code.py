import pandas as pd

# Load dataset
df = pd.read_csv("Delivery_Logistics_Cleaned.csv")

# Basic information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nAll Columns:")
for i, col in enumerate(df.columns, 1):
    print(i, col)

# ==========================================
# STEP 4 - FEATURES AND TARGET
# ==========================================

from sklearn.model_selection import train_test_split

# Target variable
target = "delivery_time_hours"

# Remove target and ID column from input features
X = df.drop(columns=[target, "delivery_id"])
y = df[target]

print("\nTarget Variable:")
print(target)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# STEP 5 - DATA PREPROCESSING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")
# ==========================================
# STEP 6 - RANDOM FOREST REGRESSION
# ==========================================

from sklearn.ensemble import RandomForestRegressor

# Create the complete machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model trained successfully!")
# ==========================================
# STEP 7 - MODEL PREDICTIONS & VALIDATION
# ==========================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours")
print(f"R² Score: {r2:.4f}")
# Save model performance
results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})

results.to_csv("model_performance.csv", index=False)

print("\nModel performance saved to model_performance.csv")
print("\n========== WEEK 4 COMPLETE ==========")