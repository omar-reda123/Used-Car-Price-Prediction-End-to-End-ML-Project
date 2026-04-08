#for location errors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#imports
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV,train_test_split
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
#6-pipeline