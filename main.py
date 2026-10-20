
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Iris Flower Prediction",
    description="Predict Iris flower species using a Deep Learning model",
    version="1.0.0",
)

# Enable CORS so the React frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Load saved model and preprocessing objects
model = load_model(BASE_DIR / "model.h5", compile=False)
scaler = joblib.load(BASE_DIR / "scaler.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width: float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width: float = Field(..., gt=0, le=10)


@app.get("/")
def home():
    return {"message": "Iris Flower Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # Keep exactly the same feature order used during training
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]])

        # Apply the existing fitted scaler
        input_scaled = scaler.transform(input_data)

        # Predict class probabilities
        probabilities = model.predict(
            input_scaled,
            verbose=0
        )[0]

        # Find the class with the highest probability
        predicted_index = int(np.argmax(probabilities))

        # Convert encoded class back to its original label
        predicted_species = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        return {
            "predicted_species": str(predicted_species),
            "confidence": round(
                float(probabilities[predicted_index]), 4
            ),
            "probabilities": {
                str(species): round(float(probability), 4)
                for species, probability in zip(
                    label_encoder.classes_,
                    probabilities
                )
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
