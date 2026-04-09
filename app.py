import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# 1. Page Configuration
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

# 2. Load Model (Cached for performance)
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_pipeline.pkl')

pipeline = load_model()

# 3. UI Design
st.title("🚗 AI Used Car Price Estimator")
st.write("Enter the car details below to get an instant AI-powered price estimate.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Car Specifications")
    brand = st.text_input("Brand (e.g., Ford, BMW, Toyota)", value="Ford")
    car_model = st.text_input("Model (e.g., Mustang, X5, Camry)", value="Mustang")
    
    # Defensive Programming: Dynamic Year Limits
    current_year = datetime.datetime.now().year
    model_year = st.number_input("Year", min_value=1990, max_value=current_year + 1, value=2015, step=1)
    
    # Defensive Programming: Hard Limits on Specs
    milage = st.number_input("Mileage (miles)", min_value=0, max_value=500000, value=50000, step=1000)
    engine_hp = st.number_input("Engine HP", min_value=50.0, max_value=1500.0, value=300.0, step=10.0)
    engine_liters = st.number_input("Engine Liters", min_value=0.8, max_value=10.0, value=5.0, step=0.1)
    engine_cylinders = st.number_input("Engine Cylinders", min_value=2, max_value=16, value=6, step=1)
    transmission_speeds = st.number_input("Transmission Speeds", min_value=1, max_value=10, value=6, step=1)

with col2:
    st.subheader("Condition & Features")
    fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Hybrid', 'Diesel', 'E85 Flex Fuel', 'Plug-In Hybrid'])
    transmission_type = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT', 'Dual Clutch'])
    ext_col = st.text_input("Exterior Color", value="Black")
    int_col = st.text_input("Interior Color", value="Black")
    accident = st.selectbox("Accident History", ['None reported', 'At least 1 accident or damage reported'])
    clean_title = st.selectbox("Clean Title", ['Yes', 'Unknown'])

# --- Soft Warnings (Smart UI) ---
if milage > 300000:
    st.warning("⚠️ Warning: This car has very high mileage. The AI will heavily depreciate its value.")

if engine_hp > 800:
    st.warning("🔥 Wow! This is supercar territory. Price predictions might vary based on brand rarity.")

st.markdown("---")

# 4. Prediction Logic
if st.button("Predict Price 💰", use_container_width=True):
    # Construct DataFrame identical to training data
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [car_model],
        'model_year': [model_year],
        'milage': [milage],
        'fuel_type': [fuel_type],
        'engine_hp': [engine_hp],
        'engine_liters': [engine_liters],
        'engine_cylinders': [engine_cylinders],       
        'ext_col': [ext_col],
        'int_col': [int_col],
        'accident': [accident],
        'clean_title': [clean_title],
        'transmission_type': [transmission_type],
        'transmission_speeds': [transmission_speeds]     
    })

    try:
        pred_log = pipeline.predict(input_data)
        pred_price = np.expm1(pred_log)[0]
        
        # --- 🛡️ Business Guardrails (طبقة حماية البيزنس) ---
        
        # 1. تسعير السيارات الخارقة (Supercars)
        if engine_hp >= 700:
            pred_price = pred_price * 2.8
        elif engine_hp >= 500:
            pred_price = pred_price * 1.5
            
        # 2. عقاب المسافات الفلكية
        if milage > 250000:
            pred_price = pred_price * 0.3
            
        # 3. عقاب الحوادث للعربيات القديمة
        if model_year < 2012 and accident == 'At least 1 accident or damage reported':
            pred_price = pred_price * 0.5
            
        # 4. الحد الأدنى لسعر الخردة
        if pred_price < 500:
            pred_price = 500.0
        
        st.success(f"### Estimated Price: ${pred_price:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"Error in prediction: {e}")