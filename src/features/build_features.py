import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger

FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_One_year",
    "Contract_Two_year",
    "PaperlessBilling_Yes",
]


def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler] | pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for column in FEATURE_COLUMNS[:3]:
        if column not in df.columns:
            raise ValueError(f"Dataset is missing required feature: {column}")
        features[column] = pd.to_numeric(df[column], errors="coerce")

    encoded_fields = {
        "Contract_One_year": ("Contract", "One year"),
        "Contract_Two_year": ("Contract", "Two year"),
        "PaperlessBilling_Yes": ("PaperlessBilling", "Yes"),
    }
    for feature, (source_column, positive_value) in encoded_fields.items():
        if feature in df.columns:
            features[feature] = pd.to_numeric(df[feature], errors="coerce")
        elif source_column in df.columns:
            normalized = df[source_column].astype(str).str.strip().str.casefold()
            features[feature] = normalized.eq(positive_value.casefold()).astype(int)
        else:
            raise ValueError(
                f"Dataset must contain either {feature} or {source_column}."
            )

    if features.isna().any().any():
        raise ValueError("Features contain missing or non-numeric values.")
    features = features[FEATURE_COLUMNS].astype(float)

    if is_train:
        fitted_scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            fitted_scaler.fit_transform(features),
            columns=FEATURE_COLUMNS,
            index=features.index,
        )
        return scaled_features, fitted_scaler

    if scaler is None:
        raise ValueError("A fitted scaler is required when is_train is False.")

    logger.debug("Transforming %d customer records.", len(features))
    return pd.DataFrame(
        scaler.transform(features),
        columns=FEATURE_COLUMNS,
        index=features.index,
    )
