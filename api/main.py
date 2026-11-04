from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.models.predict import ChurnPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "model.joblib"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "model" / "scaler.joblib"


def artifacts_available() -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (MODEL_PATH, SCALER_PATH)
    )


predictor = ChurnPredictor() if artifacts_available() else None
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for customer churn predictions.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    tenure: float = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract_One_year: int = Field(ge=0, le=1)
    Contract_Two_year: int = Field(ge=0, le=1)
    PaperlessBilling_Yes: int = Field(ge=0, le=1)


@app.get("/", status_code=status.HTTP_200_OK)
def read_root() -> dict[str, str]:
    return {"status": "online", "message": "Churn prediction service is active"}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, object]:
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "artifacts": {
            "model_weights_found": MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0,
            "scaler_found": SCALER_PATH.is_file() and SCALER_PATH.stat().st_size > 0,
            "predictor_instantiated": model_loaded,
        },
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
def get_model_info() -> dict[str, object]:
    if not artifacts_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are missing. Run the training pipeline first.",
        )

    model = joblib.load(MODEL_PATH)
    return {
        "algorithm": type(model).__name__,
        "parameters": model.get_params() if hasattr(model, "get_params") else {},
        "target_class": "Churn (Binary Classification)",
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_churn(data: CustomerData) -> dict[str, object]:
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are missing. Run the training pipeline first.",
        )

    input_df = pd.DataFrame([data.model_dump()])
    prediction, probability = predictor.predict(input_df)
    churn_prediction = int(prediction[0])
    churn_probability = float(probability[0])
    return {
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
        "retention_insight": (
            "Action required: high-risk customer profile."
            if churn_prediction == 1
            else "Stable: low churn probability."
        ),
    }
