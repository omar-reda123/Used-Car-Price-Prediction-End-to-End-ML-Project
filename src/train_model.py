# --- 1. Environment Setup ---
import sys
import os
import joblib
import numpy as np
import pandas as pd
import datetime

# Fix path for project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. Imports ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce

# Project specific files
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# --- 3. Data Loading & Preprocessing ---
print("📂 Loading and Preprocessing Data...")
df = load_data("data/processed/cleaned_data.csv")

# Drop exploration columns if they exist
columns_to_drop = ['price_log', 'milage_log']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Apply project-specific preprocessing
data = preprocess_data(df)

# Splitting Features and Target
y = data["price"]  # Target (Make sure this is Log-Scaled if you use expm1 later)
X = data.drop(columns=['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Pipeline Construction ---
print("🏗️ Building Pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('binary', OrdinalEncoder(), ['accident', 'clean_title']),
        ('low_card', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['fuel_type', 'transmission_type']),
        ('high_card', ce.TargetEncoder(), ['brand', 'model', 'ext_col', 'int_col'])
    ],
    remainder='passthrough'
)

# Base Pipeline with Random Forest
rf_pipeline = make_pipeline(
    preprocessor,
    SimpleImputer(strategy='median'),
    RandomForestRegressor(random_state=42, n_jobs=-1)
)

# --- 5. Hyperparameter Tuning (Grid Search) ---
print("🔍 Starting Grid Search (This may take a while)...")

param_grid = {
    'randomforestregressor__n_estimators': [100, 300, 500],
    'randomforestregressor__max_depth': [None, 10, 20],
    'randomforestregressor__min_samples_split': [2, 5],
    'randomforestregressor__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Use the best model found
best_pipeline = grid_search.best_estimator_

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Best CV R2 Score: {grid_search.best_score_:.4f}")

# --- 6. Prediction & Evaluation ---
print("\n📊 Evaluating Model Performance...")

# Important: Use 'best_pipeline' for prediction, not the original 'pipeline'
price_pred_log = best_pipeline.predict(X_test)

# Reverse Log Transformation (assuming target 'y' was log-scaled)
price_pred = np.expm1(price_pred_log)
y_test_real = np.expm1(y_test)

# Evaluation Metrics
r2_log = r2_score(y_test, price_pred_log)
mae = mean_absolute_error(y_test_real, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, price_pred))
mape = mean_absolute_percentage_error(y_test_real, price_pred)

print("-" * 30)
print(f"R2 Score (Log Scale): {r2_log:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"MAPE: {mape:.2%}")
print("-" * 30)

# --- 7. Saving the Final Model ---
os.makedirs("models", exist_ok=True)

# Tip: Include the date or model type in the filename for versioning
timestamp = datetime.datetime.now().strftime("%Y%m%d")
model_path = f"models/random_forest_v1_{timestamp}.pkl"

joblib.dump(best_pipeline, model_path)
print(f"🚀 Model successfully saved to: {model_path}")