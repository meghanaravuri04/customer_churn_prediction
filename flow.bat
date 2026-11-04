@echo off
setlocal
cd /d "%~dp0"

echo Creating project directories (existing files will not be changed)...
for %%D in (
    "data\raw"
    "data\processed"
    "notebooks"
    "src\data"
    "src\features"
    "src\models"
    "src\utils"
    "api"
    "frontend"
    "tests"
    "artifacts\model"
    "logs"
    ".github\workflows"
) do if not exist "%%~D" mkdir "%%~D"

echo Project directories are ready.
echo To install dependencies, run: python -m pip install -r requirements.txt
echo To download data, train, and evaluate, initialize DVC once and run: dvc repro
endlocal
