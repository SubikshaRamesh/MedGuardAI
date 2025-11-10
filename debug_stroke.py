import joblib
import pandas as pd
import numpy as np

# Load the scaler
scaler = joblib.load('models/stroke_scaler.pkl')
print('Scaler type:', type(scaler))
print('Scaler fitted:', hasattr(scaler, 'mean_'))
print('Feature names in:', hasattr(scaler, 'feature_names_in_'))

if hasattr(scaler, 'feature_names_in_'):
    print('Feature names:', scaler.feature_names_in_)
    print('Number of features expected:', len(scaler.feature_names_in_))

# Test with DataFrame
stroke_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
features = [50, 0, 0, 100, 25, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]

print('Features length:', len(features))
print('Columns length:', len(stroke_columns))

df = pd.DataFrame([features], columns=stroke_columns)
print('DataFrame shape:', df.shape)
print('DataFrame columns:', df.columns.tolist())

try:
    scaled = scaler.transform(df)
    print('Scaled shape:', scaled.shape)
    print('Success with DataFrame')
except Exception as e:
    print('DataFrame error:', e)

# Test with numpy array
try:
    features_array = np.array([features])
    scaled_array = scaler.transform(features_array)
    print('Scaled array shape:', scaled_array.shape)
    print('Success with numpy array')
except Exception as e:
    print('Numpy array error:', e)
