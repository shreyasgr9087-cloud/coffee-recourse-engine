import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from recourse_engine import BrewRecoveryEngine

# 1. Initialize App and allow CORS (so our HTML frontend can talk to it)
app = FastAPI(title="Coffee Brew AI", description="Physics-Informed AI & Recourse Engine for Coffee Extraction")

# Resolve the frontend directory relative to this file's location
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the AI Engine into memory once when the server starts
print("Loading Fickian Diffusion Brew Engine...")
engine = BrewRecoveryEngine("model.pkl")
print("Engine Ready!")

# 3. Define the Data Schemas (What the API expects to receive)
class BrewInput(BaseModel):
    roast_level: str
    grind_size_microns: float
    water_temp_c: float
    brew_time_seconds: float
    water_ratio: float

class RecourseRequest(BaseModel):
    cup: BrewInput
    mutable_features: List[str]
    target_class: str = "Balanced"
    confidence: float = 0.60

# 4. The Endpoints
@app.post("/predict")
def predict_brew(brew: BrewInput):
    """Predicts the taste profile of a coffee brew."""
    cup_dict = brew.model_dump()
    diagnosis = engine.diagnose(cup_dict)
    
    # Find the category with the highest probability
    prediction = max(diagnosis, key=diagnosis.get)
    
    return {
        "prediction": prediction,
        "confidence_scores": diagnosis
    }

@app.post("/recommend")
def recommend_fix(req: RecourseRequest):
    """Calculates the mathematical fix for a bad cup of coffee."""
    fix = engine.recommend_fix(
        cup=req.cup.model_dump(),
        mutable_features=req.mutable_features,
        target_class=req.target_class,
        confidence=req.confidence
    )
    return fix

# 5. Serve the frontend
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

# Mount static files AFTER API routes so they don't shadow /predict or /recommend
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")