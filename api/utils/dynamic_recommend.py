from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# TinyLlama model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = None
model = None

def load_tinyllama_model():
    """Load TinyLlama model and tokenizer lazily."""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading TinyLlama model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("TinyLlama model loaded successfully!")

def generate_dynamic_recommendation(patient_data: dict) -> str:
    """
    Generate personalized health recommendations using TinyLlama.

    patient_data example:
    {
        "age": 55,
        "glucose": 150,
        "blood_pressure": "140/90",
        "cholesterol": 220,
        "disease": "diabetes",
        "risk": 1
    }
    """

    try:
        # Load model if not already loaded
        load_tinyllama_model()

        # Create a structured prompt for TinyLlama
        prompt = f"<|system|>\nYou are a knowledgeable medical AI assistant providing personalized health recommendations based on patient data. Provide evidence-based, practical advice in a structured format.</s>\n<|user|>\nBased on the following patient data, provide personalized health recommendations:\n\nPatient Information:\n- Age: {patient_data.get('age', 'N/A')}\n- Glucose Level: {patient_data.get('glucose', 'N/A')}\n- Blood Pressure: {patient_data.get('blood_pressure', 'N/A')}\n- Cholesterol: {patient_data.get('cholesterol', 'N/A')}\n- Disease Type: {patient_data.get('disease', 'general').capitalize()}\n- Risk Level: {'High Risk' if patient_data.get('risk') == 1 else 'Low Risk'}\n\nPlease provide personalized recommendations in the following structured format:\n\n1. **Health Summary**: Brief assessment of current health status\n2. **Dietary Recommendations**: Specific food choices and meal planning advice\n3. **Lifestyle Modifications**: Exercise, stress management, and daily habits\n4. **Medical Monitoring**: What to watch for and when to seek professional help\n5. **Preventive Measures**: Long-term strategies to reduce disease risk\n\nKeep recommendations evidence-based, practical, and encouraging. Use clear, non-technical language.</s>\n<|assistant|>"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response (remove the prompt)
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            response = generated_text.replace(prompt, "").strip()

        # Clean up the response
        response = response.replace("<|system|>", "").replace("<|user|>", "").replace("</s>", "").strip()

        if not response or len(response) < 50:
            raise ValueError("Generated response too short or empty")

        return response

    except Exception as e:
        print(f"TinyLlama generation error: {str(e)}")
        return f"Unable to generate dynamic recommendations at this time. Error: {str(e)}. Please consult with a healthcare professional for personalized advice."
