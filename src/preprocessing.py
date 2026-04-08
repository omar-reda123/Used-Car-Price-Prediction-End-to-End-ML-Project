import pandas as pd
import numpy as np
import re

def extract_engine_features(df):
    data = df.copy() #taking copy of origin df
    #feature engineering using regex
    #if there is a problem with a value its imputed with NaN
    data['engine_liters'] = data['engine'].str.extract(r'(\d+\.\d+)(?=[Ll]|\s*[Ll]iter)').astype(float)
    data['engine_hp'] = data['engine'].str.extract(r'(\d+\.\d+)(?=[Hh][Pp])').astype(float)
    data['engine_cylinders'] = data['engine'].str.extract(r'(\d+)\s*[Cc]ylinder|[Vv]-?(\d+)|[Ss]traight\s*(\d+)').bfill(axis=1).iloc[:, 0].astype(float)
    data = data.drop(columns=['engine']) #dropping engine column
    return data
import numpy as np

def extract_transmission_features(df):
    data = df.copy() #taking copy of origin df
    #feature engineering using regex
    data['transmission_speeds'] = data['transmission'].str.extract(r'(\d+)-[Ss]peed').astype(float)
    #to lower for easier search
    trans_lower = data['transmission'].str.lower()
    
    #conditions
    is_auto = trans_lower.str.contains('auto|a/t|dual shift', na=False)
    is_manual = trans_lower.str.contains('manual|m/t', na=False)
    is_cvt = trans_lower.str.contains('cvt', na=False)

    conditions = [is_auto, is_manual, is_cvt]
    choices = ['Automatic', 'Manual', 'CVT']
    
    #apply choises based on conditions, else Other
    data['transmission_type'] = np.select(conditions, choices, default='Other')
    
    data = data.drop(columns=['transmission']) #dropping transmission column
    
    return data



def preprocess_data(df):
    """
    main function for the data pipleline
    """
    data = df.copy() #taking copy of origin df
    
    #missing values handling
    safe_condition = (data['accident'] == 'None reported') & (data['clean_title'].isna())
    data['clean_title'] = np.where(safe_condition, 'Yes', data['clean_title'].fillna('Unknown'))
    
    data['accident'] = data['accident'].fillna(data['accident'].mode()[0])
    
    data['fuel_type'] = data['fuel_type'].replace('–', 'Gasoline')

    data['fuel_type'] = data['fuel_type'].fillna(data['fuel_type'].mode()[0])
    
    #feature enfineering
    data = extract_engine_features(data)
    data = extract_transmission_features(data)
    
    #Log Transformation
    data['milage'] = np.log1p(data['milage'])
    
    if 'price' in data.columns:
        data['price'] = np.log1p(data['price'])
        
    return data