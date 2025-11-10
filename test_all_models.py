import numpy as np
from api.utils.load_models import diabetes_model, diabetes_scaler, heart_model, heart_scaler, stroke_model, stroke_scaler
import pandas as pd

def test_diabetes():
    # Sample features for diabetes (8 features)
    features = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    scaled = diabetes_scaler.transform(features)
    pred = diabetes_model.predict(scaled)
    print(f"Diabetes model prediction: {pred[0]}")
    return True

def test_heart():
    # Sample features for heart (11 features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
    features = np.array([[55, 1, 170, 80, 120, 80, 1, 1, 0, 0, 1]])
    scaled = heart_scaler.transform(features)
    pred = heart_model.predict(scaled)
    print(f"Heart model prediction: {pred[0]}")
    return True

def test_stroke():
    # Sample features for stroke (16 features)
    stroke_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
    features = [50, 0, 0, 100, 25, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # Sample data
    df = pd.DataFrame([features], columns=stroke_columns)
    scaled = stroke_scaler.transform(df)
    pred = stroke_model.predict(scaled)
    print(f"Stroke model prediction: {pred[0]}")
    return True

if __name__ == "__main__":
    try:
        print("Testing Diabetes Model...")
        test_diabetes()
        print("Diabetes model working.")
    except Exception as e:
        print(f"Diabetes model error: {e}")

    try:
        print("Testing Heart Model...")
        test_heart()
        print("Heart model working.")
    except Exception as e:
        print(f"Heart model error: {e}")

    try:
        print("Testing Stroke Model...")
        test_stroke()
        print("Stroke model working.")
    except Exception as e:
        print(f"Stroke model error: {e}")

    print("All tests completed.")
