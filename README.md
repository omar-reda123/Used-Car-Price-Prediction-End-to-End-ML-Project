# 🚗 AI Used Car Price Estimator (v1.0.0)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Frontend](https://img.shields.io/badge/UI-Streamlit-red.svg)

## 📌 Project Overview
An end-to-end Machine Learning system designed to predict the price of used cars with high accuracy. Unlike standard tutorial models, this project incorporates **Business Guardrails** to handle edge cases (like extreme supercars or scrap vehicles), bridging the gap between raw algorithmic predictions and real-world market logic.

## 📖 The Engineering Journey: Building an End-to-End System
This project was built from scratch to simulate a real-world data science lifecycle. Here is the step-by-step pipeline:

1. **Data Collection & Structural Fixes 🧹:** The dataset had a critical structural bug right out of the gate: car brands with two or more words were parsed incorrectly during data collection (the first word was stored as `brand`, and the rest spilled over into the `model` column). I engineered a custom script to detect and fix this shifting issue before proceeding. Afterwards, I handled missing values (Median Imputation) and removed duplicates.
2. **Exploratory Data Analysis (EDA) 📊:** Analyzed feature relationships and discovered a right-skewed price distribution. Applied a `Log Transformation (np.log1p)` to normalize the target variable, significantly boosting model stability.
3. **Preprocessing & Feature Engineering ⚙️:** Extracted valuable features like `engine_cylinders` and `transmission_speeds`. To handle high-cardinality categorical data (like Car Brands and Models) without exploding dimensionality, I utilized `TargetEncoder` integrated within a Scikit-learn Pipeline.
4. **Model Training & Critical Evaluation 🧠:** Trained an **XGBoost Regressor** as the primary engine. During evaluation, I noticed a significant gap between the Mean Absolute Error (MAE) and the Root Mean Squared Error (RMSE). This is a strong indicator that extreme outliers (hyper-expensive supercars or severely damaged vehicles) are still punishing the model. Acknowledging this behavior was a key insight for future versions.
5. **Business Logic & Guardrails 🛡️:** Since pure ML models fail on extreme edge cases, I implemented custom "Business Guardrails" during inference. This logic intelligently depreciates scrap cars and appreciates supercars based on HP and mileage, mimicking human market expertise and mitigating some of the outlier effects.
6. **Deployment & UI 💻:** Developed an interactive web application using **Streamlit**. Implemented `Defensive Programming` in the frontend to restrict illogical user inputs (e.g., entering a year in the distant future) and provide real-time UI warnings.

## ✨ Key Features
* **Robust ML Engine:** Powered by a baseline **XGBoost Regressor**.
* **Smart Business Guardrails:** Custom inference logic intercepts the model's output to correctly price extreme outliers.
* **Defensive Frontend:** Interactive UI with strict input validations to prevent nonsensical predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy, Scikit-learn, Category Encoders
* **Modeling:** XGBoost Regressor
* **Deployment & UI:** Streamlit, Joblib

## 📂 Project Structure
```text
📦 Used-Car-Price-Prediction-End-to-End-ML-Project
 ┣ 📂 data
 ┃ ┣ 📂 raw             # Original dataset
 ┃ ┗ 📂 processed       # Cleaned dataset ready for training
 ┣ 📂 models            # Saved ML models (e.g., xgboost_pipeline.pkl)
 ┣ 📂 notebooks         # Jupyter notebooks for EDA and initial experiments
 ┣ 📂 src               # Source code for modular scripts
 ┃ ┣ 📜 data_loader.py
 ┃ ┣ 📜 preprocessing.py
 ┃ ┗ 📜 train_model.py
 ┣ 📜 app.py            # Streamlit frontend application
 ┣ 📜 requirements.txt  # Project dependencies
 ┗ 📜 README.md

 🚀 How to Run Locally
1. Clone the repository:
git clone [https://github.com/omar-reda123/Used-Car-Price-Prediction-End-to-End-ML-Project.git](https://github.com/omar-reda123/Used-Car-Price-Prediction-End-to-End-ML-Project.git)
cd Used-Car-Price-Prediction-End-to-End-ML-Project

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit App:
streamlit run app.py

🔮 Future Enhancements (v2.0)
[1] Implement GridSearchCV for rigorous hyperparameter tuning to squeeze out maximum model performance.

[2] Apply advanced Outlier Detection & Handling techniques to close the gap between MAE and RMSE.

[3] Migrate backend to FastAPI to decouple the ML logic from the UI.

[4] Implement a Database (e.g., PostgreSQL) to log user queries for future retraining.

[5] Explore CatBoost to optimize categorical feature handling further.