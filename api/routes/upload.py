from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
import pytesseract
from PIL import Image
import io
from api.utils.nlp_processor import extract_health_metrics, map_to_model_features
from api.utils.load_models import diabetes_model, heart_model, stroke_model
from api.utils.load_models import diabetes_scaler, heart_scaler, stroke_scaler
from api.utils.recommend import generate_recommendation
import numpy as np

router = APIRouter()

def extract_text_from_pdf(file: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(file: bytes) -> str:
    """Extract text from image using Tesseract OCR."""
    image = Image.open(io.BytesIO(file))
    text = pytesseract.image_to_string(image)
    return text

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF or image), extract health metrics using NLP,
    predict disease risks, and provide recommendations.
    """
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png", "image/jpg", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, image, or text file.")

    try:
        # Read file content
        file_content = await file.read()

        # Extract text based on file type
        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(file_content)
        elif file.content_type == "text/plain":
            text = file_content.decode('utf-8')
        else:
            text = extract_text_from_image(file_content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

        # Extract health metrics using NLP
        extracted_metrics = extract_health_metrics(text)

        if not extracted_metrics:
            return JSONResponse(content={
                "message": "No health metrics could be extracted from the document.",
                "extracted_text": text[:500],  # First 500 chars for debugging
                "predictions": {},
                "recommendations": {}
            })

        # Determine which model to use based on available metrics
        predictions = {}
        recommendations = {}

        # Check for diabetes features
        diabetes_features = map_to_model_features(extracted_metrics, "diabetes")
        if any(diabetes_features):
            scaled = diabetes_scaler.transform(np.array(diabetes_features).reshape(1, -1))
            pred = diabetes_model.predict(scaled)[0]
            predictions["diabetes"] = int(pred)
            recommendations["diabetes"] = generate_recommendation("diabetes", pred)

        # Check for heart features
        heart_features = map_to_model_features(extracted_metrics, "heart")
        if any(heart_features):
            scaled = heart_scaler.transform(np.array(heart_features).reshape(1, -1))
            pred = heart_model.predict(scaled)[0]
            predictions["heart"] = int(pred)
            recommendations["heart"] = generate_recommendation("heart", pred)

        # Check for stroke features
        stroke_features = map_to_model_features(extracted_metrics, "stroke")
        if any(stroke_features):
            scaled = stroke_scaler.transform(np.array(stroke_features).reshape(1, -1))
            pred = stroke_model.predict(scaled)[0]
            predictions["stroke"] = int(pred)
            recommendations["stroke"] = generate_recommendation("stroke", pred)

        return JSONResponse(content={
            "message": "File processed successfully.",
            "extracted_metrics": extracted_metrics,
            "predictions": predictions,
            "recommendations": recommendations
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
