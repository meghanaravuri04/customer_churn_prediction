import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.features.build_features import FEATURE_COLUMNS, build_features


@pytest.fixture
def sample_raw_data(tmp_path):
    rows = []
    for index in range(20):
        rows.append(
            {
                "customerID": f"customer-{index}",
                "tenure": index + 1,
                "MonthlyCharges": 50 + index,
                "TotalCharges": 100 + index * 10,
                "Contract": "One year" if index % 2 else "Month-to-month",
                "PaperlessBilling": "Yes" if index % 2 else "No",
                "Churn": "Yes" if index % 2 else "No",
            }
        )
    raw_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    return raw_path


def test_preprocess_pipeline_splits_and_encodes_target(sample_raw_data, tmp_path):
    processed_dir = tmp_path / "processed"
    preprocess_data(str(sample_raw_data), str(processed_dir))

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    assert len(train_df) + len(test_df) == 20
    assert set(train_df["Churn"].unique()) <= {0, 1}
    assert set(test_df["Churn"].unique()) <= {0, 1}


def test_build_features_uses_same_six_columns_for_training_and_inference():
    df = pd.DataFrame(
        {
            "tenure": [12, 24],
            "MonthlyCharges": [70.0, 80.0],
            "TotalCharges": [840.0, 1920.0],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
        }
    )
    train_features, scaler = build_features(df, is_train=True)
    inference_features = build_features(df, is_train=False, scaler=scaler)

    assert train_features.columns.tolist() == FEATURE_COLUMNS
    assert inference_features.columns.tolist() == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(train_features, inference_features)


def test_build_features_requires_scaler_for_inference():
    df = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.0],
            "TotalCharges": [840.0],
            "Contract_One_year": [0],
            "Contract_Two_year": [0],
            "PaperlessBilling_Yes": [1],
        }
    )
    with pytest.raises(ValueError, match="fitted scaler"):
        build_features(df, is_train=False)
