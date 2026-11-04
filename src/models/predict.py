from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChurnPredictor:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        artifact_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "artifacts" / "model"
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.scaler = joblib.load(artifact_dir / "scaler.joblib")

    def predict(
        self,
        input_data: pd.DataFrame,
    ) -> tuple[list[int], list[float]]:
        features = build_features(
            input_data,
            is_train=False,
            scaler=self.scaler,
        )
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        return predictions.tolist(), probabilities.tolist()
