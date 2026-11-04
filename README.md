# Customer Churn Prediction

This project downloads the IBM Telco Customer Churn dataset, prepares features,
compares classifiers, saves the selected model, and serves predictions through
FastAPI and Streamlit.

## Windows setup

Use Python 3.13 (the same version used by the CI and Docker images).

From the project directory, create and activate a virtual environment, then
install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`flow.bat` only creates missing project directories; it does not overwrite
project files. It is safe to run again:

```powershell
.\flow.bat
```

## Build the data and model pipeline

This checkout is already initialized with DVC. Run all stages with the command
below. In a fresh copy, initialize DVC once first: use `dvc init --no-scm` if
the directory is not a Git repository, or `dvc init` otherwise. The download
stage uses KaggleHub to fetch the public dataset and requires internet access.

```powershell
dvc repro
```

The pipeline downloads the dataset, creates stratified train/test splits, trains
and selects a model using a validation split, saves the model and scaler under
`artifacts/model/`, and evaluates the holdout set.

Run the test suite with:

```powershell
python -m pytest tests -v
```

## Run the services

After the pipeline has generated model artifacts, run the API in one terminal:

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Run the Streamlit interface in another terminal:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run frontend/app.py
```

The API is available at `http://127.0.0.1:8000`; its interactive docs are at
`http://127.0.0.1:8000/docs`. Streamlit is available at
`http://localhost:8501`.

Alternatively, start both services with Docker Compose:

```powershell
docker compose up --build
```
