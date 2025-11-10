from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import pandas as pd
from api.utils.load_models import diabetes_model, heart_model, stroke_model
from api.utils.load_models import diabetes_scaler, heart_scaler, stroke_scaler
from api.utils.recommend import generate_recommendation

router = APIRouter()

class InputData(BaseModel):
    model_type: str
    features: list

class CardioInput(BaseModel):
    features: list
@router.post("/predict")
def predict(data: InputData):
    model_type = data.model_type.lower()
    features = np.array(data.features).reshape(1, -1)

    if model_type == "diabetes":
        scaled = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(scaled)
        recommendation = generate_recommendation("diabetes", int(prediction[0]))
    elif model_type == "heart":
        try:
            scaled = heart_scaler.transform(features)
            prediction = heart_model.predict(scaled)
            recommendation = generate_recommendation("heart", int(prediction[0]))
        except ValueError as e:
            return {"error": f"Feature mismatch: {str(e)}"}
    elif model_type == "stroke":
        try:
            # Stroke scaler expects DataFrame with feature names
            stroke_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
            df = pd.DataFrame(features, columns=stroke_columns)
            scaled = stroke_scaler.transform(df)
            prediction = stroke_model.predict(scaled)
            recommendation = generate_recommendation("stroke", int(prediction[0]))
        except Exception as e:
            return {"error": f"Stroke prediction error: {str(e)}"}
    else:
        return {"error": "Invalid model type"}

    return {
        "prediction": int(prediction[0]),
        "recommendation": recommendation
    }

@router.post("/predict/cardio")
def predict_cardio(data: CardioInput):
    features = np.array(data.features).reshape(1, -1)
    try:
        scaled = heart_scaler.transform(features)
        prediction = heart_model.predict(scaled)
        message = "Cardiovascular disease likely" if prediction[0] == 1 else "Low cardiovascular disease risk"
        recommendation = generate_recommendation("heart", int(prediction[0]))
        return {
            "prediction": int(prediction[0]),
            "message": message,
            "model_used": "heart",
            "recommendation": recommendation
        }
    except ValueError as e:
        return {"error": f"Feature mismatch: {str(e)}"}
