from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaperlessBilling",
    "Churn",
}


def preprocess_data(raw_data_path: str, processed_dir: str) -> None:
    raw_path = Path(raw_data_path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)

    df = pd.read_csv(raw_path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    churn = df["Churn"].astype(str).str.strip().str.lower()
    churn_mapping = {"yes": 1, "no": 0, "1": 1, "0": 0}
    df["Churn"] = churn.map(churn_mapping)
    df = df.dropna(
        subset=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "PaperlessBilling",
            "Churn",
        ]
    )
    df["Churn"] = df["Churn"].astype(int)
    if df.empty:
        raise ValueError("Dataset has no usable rows after cleaning.")

    prepare_params = params["prepare"]
    train_df, test_df = train_test_split(
        df,
        test_size=prepare_params["test_size"],
        random_state=prepare_params["random_state"],
        stratify=df["Churn"] if df["Churn"].nunique() > 1 else None,
    )

    output_dir = Path(processed_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    logger.info(
        "Preprocessing complete: %s training rows and %s test rows.",
        len(train_df),
        len(test_df),
    )


if __name__ == "__main__":
    preprocess_data("data/raw/dataset.csv", "data/processed")
