import pandas as pd
from api.utils.load_models import diabetes_model, diabetes_scaler
import numpy as np

# Load diabetes data
data = pd.read_csv('data/diabetes.csv')

# Get a positive sample (Outcome=1)
positive_sample = data[data['Outcome'] == 1].iloc[0]
features = positive_sample[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
print('Positive sample features:', features)
print('Expected outcome: 1')

# Scale and predict
scaled = diabetes_scaler.transform([features])
prediction = diabetes_model.predict(scaled)
print('Model prediction:', prediction[0])

# Also test with a negative sample
negative_sample = data[data['Outcome'] == 0].iloc[0]
features_neg = negative_sample[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
print('Negative sample features:', features_neg)
print('Expected outcome: 0')

scaled_neg = diabetes_scaler.transform([features_neg])
prediction_neg = diabetes_model.predict(scaled_neg)
print('Model prediction for negative:', prediction_neg[0])
