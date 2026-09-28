#!/usr/bin/env bash
# DelayShield - train the model (if needed) and launch the dashboard.
set -e
cd "$(dirname "$0")"

if [ ! -f models/delay_model.pkl ]; then
  echo "[DelayShield] No trained model found - running the offline pipeline..."
  python src/train.py
fi

echo "[DelayShield] Starting the dashboard on http://localhost:8501"
streamlit run dashboard/app.py
