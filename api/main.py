from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.predict import router as predict_router
from api.routes.upload import router as upload_router

app = FastAPI(title="MedGuardAI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Welcome to MedGuardAI API!"}
