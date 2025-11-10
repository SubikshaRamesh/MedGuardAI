import re
import spacy
from typing import Dict, List, Any

# Load spaCy model (ensure 'en_core_web_sm' is downloaded)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def extract_health_metrics(text: str) -> Dict[str, Any]:
    """
    Extract health metrics from text using regex and spaCy NER.
    Maps to features for diabetes, heart, stroke models.
    """
    extracted = {}

    # Define regex patterns for common health metrics
    patterns = {
        "glucose": r"glucose[:\s]*(\d+(?:\.\d+)?)",
        "blood_pressure_systolic": r"(?:blood pressure|BP)[:\s]*(\d+)",
        "blood_pressure_diastolic": r"(?:blood pressure|BP)[:\s]*\d+[/\s](\d+)",
        "cholesterol": r"cholesterol[:\s]*(\d+(?:\.\d+)?)",
        "hdl": r"HDL[:\s]*(\d+(?:\.\d+)?)",
        "ldl": r"LDL[:\s]*(\d+(?:\.\d+)?)",
        "triglycerides": r"triglycerides[:\s]*(\d+(?:\.\d+)?)",
        "bmi": r"BMI[:\s]*(\d+(?:\.\d+)?)",
        "age": r"age[:\s]*(\d+)",
        "height": r"height[:\s]*(\d+(?:\.\d+)?)",
        "weight": r"weight[:\s]*(\d+(?:\.\d+)?)",
        "pregnancies": r"pregnanc(?:y|ies)[:\s]*(\d+)",
        "skin_thickness": r"skin thickness[:\s]*(\d+(?:\.\d+)?)",
        "insulin": r"insulin[:\s]*(\d+(?:\.\d+)?)",
        "diabetes_pedigree": r"diabetes pedigree[:\s]*(\d+(?:\.\d+)?)",
        "gender": r"gender[:\s]*(male|female|man|woman)",
        "ap_hi": r"(?:systolic|ap_hi)[:\s]*(\d+)",
        "ap_lo": r"(?:diastolic|ap_lo)[:\s]*(\d+)",
        "cholesterol_level": r"cholesterol[:\s]*(\d+)",
        "gluc": r"glucose[:\s]*(\d+)",
        "smoke": r"smok(?:e|ing)[:\s]*(yes|no|1|0)",
        "alco": r"alcohol[:\s]*(yes|no|1|0)",
        "active": r"active[:\s]*(yes|no|1|0)",
        "hypertension": r"hypertension[:\s]*(yes|no|1|0)",
        "heart_disease": r"heart disease[:\s]*(yes|no|1|0)",
        "ever_married": r"(?:ever married|married)[:\s]*(yes|no|1|0)",
        "work_type": r"work type[:\s]*(private|self-employed|govt_job|children|never_worked)",
        "residence_type": r"residence[:\s]*(urban|rural)",
        "avg_glucose_level": r"avg glucose[:\s]*(\d+(?:\.\d+)?)",
        "bmi_stroke": r"BMI[:\s]*(\d+(?:\.\d+)?)"
    }

    text_lower = text.lower()

    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key in ["gender", "smoke", "alco", "active", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type"]:
                if value in ["yes", "male", "man", "private", "self-employed", "govt_job", "children", "never_worked", "urban"]:
                    extracted[key] = 1
                elif value in ["no", "female", "woman", "rural"]:
                    extracted[key] = 0
                else:
                    extracted[key] = int(value) if value.isdigit() else value
            else:
                try:
                    extracted[key] = float(value)
                except ValueError:
                    extracted[key] = value

    # Use spaCy for additional entity recognition if model is loaded
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and "age" not in extracted:
                # Try to extract age near person name (simple heuristic)
                age_match = re.search(r"(\d{1,3})", ent.sent.text)
                if age_match:
                    extracted["age"] = int(age_match.group(1))

    return extracted

def map_to_model_features(extracted: Dict[str, Any], model_type: str) -> List[float]:
    """
    Map extracted metrics to the required feature list for each model.
    Returns a list of features in the correct order.
    """
    if model_type == "diabetes":
        # Features: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age
        features = [
            extracted.get("pregnancies", 0),
            extracted.get("glucose", 0),
            extracted.get("blood_pressure_systolic", 0),
            extracted.get("skin_thickness", 0),
            extracted.get("insulin", 0),
            extracted.get("bmi", 0),
            extracted.get("diabetes_pedigree", 0),
            extracted.get("age", 0)
        ]
    elif model_type == "heart":
        # Features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        features = [
            extracted.get("age", 0),
            extracted.get("gender", 0),  # 1 for male, 0 for female
            extracted.get("height", 0),
            extracted.get("weight", 0),
            extracted.get("ap_hi", extracted.get("blood_pressure_systolic", 0)),
            extracted.get("ap_lo", extracted.get("blood_pressure_diastolic", 0)),
            extracted.get("cholesterol_level", extracted.get("cholesterol", 0)),
            extracted.get("gluc", extracted.get("glucose", 0)),
            extracted.get("smoke", 0),
            extracted.get("alco", 0),
            extracted.get("active", 0)
        ]
    elif model_type == "stroke":
        # Features: age, hypertension, heart_disease, avg_glucose_level, bmi, gender_Male, gender_Other, ever_married_Yes, work_type_Never_worked, work_type_Private, work_type_Self-employed, work_type_children, Residence_type_Urban, smoking_status_formerly smoked, smoking_status_never smoked, smoking_status_smokes
        features = [
            extracted.get("age", 0),
            extracted.get("hypertension", 0),
            extracted.get("heart_disease", 0),
            extracted.get("avg_glucose_level", extracted.get("glucose", 0)),
            extracted.get("bmi_stroke", extracted.get("bmi", 0)),
            1 if extracted.get("gender", 0) == 1 else 0,  # gender_Male
            0,  # gender_Other (assuming not present in text)
            1 if extracted.get("ever_married", 0) == 1 else 0,  # ever_married_Yes
            1 if extracted.get("work_type") == "never_worked" else 0,  # work_type_Never_worked
            1 if extracted.get("work_type") == "private" else 0,  # work_type_Private
            1 if extracted.get("work_type") == "self-employed" else 0,  # work_type_Self-employed
            1 if extracted.get("work_type") == "children" else 0,  # work_type_children
            1 if extracted.get("residence_type", 0) == 1 else 0,  # Residence_type_Urban
            1 if extracted.get("smoke") == "formerly smoked" else 0,  # smoking_status_formerly smoked
            1 if extracted.get("smoke") == "never smoked" else 0,  # smoking_status_never smoked
            1 if extracted.get("smoke") == "smokes" else 0   # smoking_status_smokes
        ]
    else:
        features = []

    return features
