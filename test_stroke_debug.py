import numpy as np
import pandas as pd
from api.utils.load_models import stroke_scaler, stroke_model
from api.utils.recommend import generate_recommendation

features = np.array([50, 0, 0, 100, 25, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]).reshape(1, -1)
print('Features shape:', features.shape)
print('Features:', features)

try:
    scaled = stroke_scaler.transform(features)
    prediction = stroke_model.predict(scaled)
    print('Prediction:', prediction[0])
    recommendation = generate_recommendation('stroke', int(prediction[0]))
    print('Recommendation:', recommendation)
except Exception as e:
    print('Error:', str(e))

# Try with DataFrame
stroke_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
df = pd.DataFrame(features, columns=stroke_columns)
print('DataFrame shape:', df.shape)
print('DataFrame columns:', df.columns.tolist())

try:
    scaled_df = stroke_scaler.transform(df)
    prediction_df = stroke_model.predict(scaled_df)
    print('Prediction with DataFrame:', prediction_df[0])
    recommendation_df = generate_recommendation('stroke', int(prediction_df[0]))
    print('Recommendation with DataFrame:', recommendation_df)
except Exception as e:
    print('Error with DataFrame:', str(e))
