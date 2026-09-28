@echo off
REM DelayShield - train the model (if needed) and launch the dashboard.
cd /d "%~dp0"

if not exist "models\delay_model.pkl" (
    echo [DelayShield] No trained model found - running the offline pipeline...
    python src\train.py || exit /b 1
)

echo [DelayShield] Starting the dashboard on http://localhost:8501
streamlit run dashboard\app.py
