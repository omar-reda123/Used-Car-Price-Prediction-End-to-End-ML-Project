#for location errors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#imports
#manpulations,model file
import joblib
import numpy as np
import pandas as pd
#models
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#evaluation metrics
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
#modeling helpers
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV,train_test_split
#encoding and imputation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.impute import SimpleImputer
#project files
from src.data_loader import load_data
from src.preprocessing import preprocess_data

#1-reading data after cleaning
df=load_data("data/processed/cleaned_data.csv")
#2-dropping price_log and milage_log(already will be done in preprocessing, I did them in EDA just to explore)
df = df.drop(columns=['price_log', 'milage_log'])
#3-preprocessing
data=preprocess_data(df)
#4-splitting features from target
y=data["price"] #target
X=data.drop(columns=['price']) #features
#5-train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#6-encoder
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', OrdinalEncoder(), ['accident', 'clean_title']),
        ('low_card', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['fuel_type', 'transmission_type']),
        ('high_card', ce.TargetEncoder(), ['brand', 'model', 'ext_col', 'int_col'])
    ],
    remainder='passthrough' #numerical features remain without change
)

# 7- Pipeline
pipeline = make_pipeline(
    preprocessor,
    SimpleImputer(strategy='median'), #if NaN>> replace with column median
    #StandardScaler(),
    XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=42, n_jobs=-1)
)

pipeline.fit(X_train,y_train)
price_pred_log=pipeline.predict(X_test)
#reversing log int exp
price_pred=np.expm1(price_pred_log)
y_test_real = np.expm1(y_test)
#8-evaluation
print("--- Model Evaluation ---")

# 1. R2 on Log Scale (Model Learning Power)
r2_log = r2_score(y_true=y_test, y_pred=price_pred_log)
print(f"R2 Score (Log Scale): {r2_log:.4f}")

# 2. Mean Absolute Error (Business Metric)
mae = mean_absolute_error(y_true=y_test_real, y_pred=price_pred)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

# 3. Root Mean Squared Error (Outlier Catcher)
rmse = np.sqrt(mean_squared_error(y_true=y_test_real, y_pred=price_pred))
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# 4. Mean Absolute Percentage Error (Fairness Metric)
mape = mean_absolute_percentage_error(y_true=y_test_real, y_pred=price_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")