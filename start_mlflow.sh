#!/bin/bash
# Start MLflow UI with SQLite backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting MLflow UI on http://localhost:5000"
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
