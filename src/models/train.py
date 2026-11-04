import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import build_features
from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_and_track_models(processed_dir: str, model_dir: str) -> None:
    data_dir = Path(processed_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    train_df = pd.read_csv(data_dir / "train.csv")
    y = train_df["Churn"].astype(int)
    config_path = PROJECT_ROOT / "params.yaml"
    with config_path.open(encoding="utf-8") as params_file:
        config = yaml.safe_load(params_file)
    validation_size = config["train"]["validation_size"]

    has_stratifiable_classes = (
        y.nunique() > 1
        and y.value_counts().min() >= 2
        and int(len(train_df) * validation_size) >= y.nunique()
    )
    if len(train_df) >= 5 and has_stratifiable_classes:
        fit_df, validation_df = train_test_split(
            train_df,
            test_size=validation_size,
            random_state=config["prepare"]["random_state"],
            stratify=y,
        )
    else:
        fit_df = train_df
        validation_df = train_df

    y_fit = fit_df["Churn"].astype(int)
    y_validation = validation_df["Churn"].astype(int)
    x_fit, fit_scaler = build_features(fit_df, is_train=True)
    x_validation = build_features(
        validation_df,
        is_train=False,
        scaler=fit_scaler,
    )

    train_params = config["train"]
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            max_iter=1000,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=42,
        ),
    }

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Customer_Churn_Evaluation")
    best_model_name = ""
    best_f1 = -1.0
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(x_fit, y_fit)
            predictions = model.predict(x_validation)
            validation_f1 = f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
            mlflow.log_param("estimator", model_name)
            mlflow.log_metric("validation_f1_score", validation_f1)
            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", train_params["n_estimators"])
                mlflow.log_param("max_depth", train_params["max_depth"])
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            logger.info("%s validation F1: %.4f", model_name, validation_f1)
            if validation_f1 > best_f1:
                best_model_name = model_name
                best_f1 = validation_f1

    full_x, full_scaler = build_features(train_df, is_train=True)
    best_model = models[best_model_name]
    best_model.fit(full_x, y)

    output_dir = Path(model_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "model.joblib")
    joblib.dump(full_scaler, output_dir / "scaler.joblib")
    logger.info(
        "Saved %s with validation F1 %.4f and its feature scaler to %s.",
        best_model_name,
        best_f1,
        output_dir,
    )


if __name__ == "__main__":
    train_and_track_models("data/processed", "artifacts/model")
