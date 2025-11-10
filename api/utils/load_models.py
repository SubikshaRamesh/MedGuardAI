import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

# ✅ Load ML models
diabetes_model = load_model("diabetes_model.pkl")
heart_model = load_model("heart_model.pkl")
stroke_model = load_model("stroke_model.pkl")

# ✅ Load corresponding scalers
diabetes_scaler = load_model("diabetes_scaler.pkl")
heart_scaler = load_model("heart_scaler.pkl")
stroke_scaler = load_model("stroke_scaler.pkl")

# print("✅ All models and scalers loaded successfully!")
