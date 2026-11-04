from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.features.build_features import build_features
from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_model(processed_dir: str, model_dir: str) -> dict[str, object]:
    data_dir = Path(processed_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    artifacts_dir = Path(model_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = PROJECT_ROOT / artifacts_dir

    test_df = pd.read_csv(data_dir / "test.csv")
    y_test = test_df["Churn"].astype(int)
    model = joblib.load(artifacts_dir / "model.joblib")
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    x_test = build_features(test_df, is_train=False, scaler=scaler)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    logger.info("Holdout accuracy: %.4f", accuracy)
    print(f"Accuracy: {accuracy:.4f}\nClassification Report:\n{report}")
    return {"accuracy": accuracy, "classification_report": report}


if __name__ == "__main__":
    evaluate_model("data/processed", "artifacts/model")
