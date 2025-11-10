def generate_recommendation(disease, prediction):
    if disease == "diabetes":
        if prediction == 1:
            return {
                "health_condition": "High risk of diabetes",
                "healthcare_tips": [
                    "Schedule regular check-ups with an endocrinologist.",
                    "Monitor blood glucose levels daily.",
                    "Consult a dietitian for personalized meal plans."
                ],
                "food_suggestions": [
                    "Focus on low glycemic index foods like whole grains, vegetables, and legumes.",
                    "Include lean proteins such as fish, chicken, and tofu.",
                    "Limit sugary drinks and processed foods; opt for water and herbal teas."
                ],
                "tips_to_reduce_problem": [
                    "Aim for 150 minutes of moderate aerobic activity per week.",
                    "Maintain a healthy weight; lose 5-10% of body weight if overweight.",
                    "Quit smoking and limit alcohol intake.",
                    "Practice stress management techniques like meditation or yoga."
                ]
            }
        else:
            return {
                "health_condition": "Low risk of diabetes",
                "healthcare_tips": [
                    "Continue regular health screenings.",
                    "Maintain healthy lifestyle habits."
                ],
                "food_suggestions": [
                    "Eat a balanced diet rich in fruits, vegetables, and whole grains.",
                    "Stay hydrated and enjoy moderate portions."
                ],
                "tips_to_reduce_problem": [
                    "Keep up regular physical activity.",
                    "Monitor weight and blood pressure periodically."
                ]
            }
    elif disease == "heart":
        if prediction == 1:
            return {
                "health_condition": "High risk of cardiovascular disease",
                "healthcare_tips": [
                    "Consult a cardiologist immediately for further evaluation.",
                    "Get regular ECG, echocardiogram, and stress tests.",
                    "Monitor blood pressure and cholesterol levels closely."
                ],
                "food_suggestions": [
                    "Adopt a Mediterranean diet: emphasize olive oil, nuts, fish, fruits, and vegetables.",
                    "Reduce salt intake to less than 2,300 mg per day.",
                    "Choose lean meats, low-fat dairy, and whole grains."
                ],
                "tips_to_reduce_problem": [
                    "Engage in at least 150 minutes of moderate exercise weekly.",
                    "Quit smoking and avoid secondhand smoke.",
                    "Manage stress through relaxation techniques.",
                    "Control diabetes and hypertension if present."
                ]
            }
        else:
            return {
                "health_condition": "Low risk of cardiovascular disease",
                "healthcare_tips": [
                    "Continue annual check-ups and screenings.",
                    "Maintain healthy habits to prevent future risks."
                ],
                "food_suggestions": [
                    "Eat heart-healthy foods like fatty fish, nuts, and leafy greens.",
                    "Limit saturated fats and trans fats."
                ],
                "tips_to_reduce_problem": [
                    "Stay active with regular walking or cycling.",
                    "Monitor blood pressure and cholesterol."
                ]
            }
    elif disease == "stroke":
        if prediction == 1:
            return {
                "health_condition": "High risk of stroke",
                "healthcare_tips": [
                    "See a neurologist for risk assessment and preventive measures.",
                    "Monitor for symptoms like sudden numbness or confusion.",
                    "Get regular carotid artery ultrasounds if advised."
                ],
                "food_suggestions": [
                    "Follow DASH diet: focus on fruits, vegetables, whole grains, and low-fat dairy.",
                    "Reduce sodium to 1,500 mg daily.",
                    "Include potassium-rich foods like bananas and spinach."
                ],
                "tips_to_reduce_problem": [
                    "Control hypertension, diabetes, and atrial fibrillation.",
                    "Exercise regularly: aim for 150 minutes of moderate activity.",
                    "Stop smoking and limit alcohol.",
                    "Maintain healthy weight and manage cholesterol."
                ]
            }
        else:
            return {
                "health_condition": "Low risk of stroke",
                "healthcare_tips": [
                    "Keep up with routine health exams.",
                    "Stay informed about stroke warning signs."
                ],
                "food_suggestions": [
                    "Eat a balanced diet with plenty of antioxidants from fruits and vegetables.",
                    "Choose healthy fats from fish and nuts."
                ],
                "tips_to_reduce_problem": [
                    "Exercise to maintain cardiovascular health.",
                    "Avoid smoking and excessive alcohol."
                ]
            }
    else:
        return {
            "health_condition": "Unknown",
            "healthcare_tips": ["Consult a healthcare professional for personalized advice."],
            "food_suggestions": ["Maintain a balanced diet."],
            "tips_to_reduce_problem": ["Follow general health guidelines."]
        }
