import shutil
from pathlib import Path

import kagglehub


def download_dataset():
    """Download the IBM Telco Customer Churn dataset and copy its CSV."""

    project_root = Path(__file__).resolve().parents[2]
    destination = project_root / "data" / "raw" / "dataset.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        "blastchar/telco-customer-churn"
    )

    print("Downloaded dataset location:", dataset_path)

    csv_files = sorted(Path(dataset_path).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")

    source = csv_files[0]
    shutil.copy2(source, destination)

    print("Dataset copied successfully!")
    print("Saved location:", destination)


if __name__ == "__main__":
    download_dataset()