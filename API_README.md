# Drug Classification API

FastAPI-based REST API for predicting drug prescriptions based on patient characteristics using a trained Random Forest model.

## Features

- **Single Prediction**: Predict drug for one patient
- **Batch Prediction**: Predict drugs for multiple patients
- **Model Information**: Get details about the loaded model
- **Health Check**: Verify API and model status
- **Interactive Documentation**: Swagger UI and ReDoc

## Installation

1. Install dependencies:
```bash
make install
```

2. Train the model (if not already trained):
```bash
make pipeline
```

## Running the API

### Start the server:
```bash
make serve
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns server health status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 3. Model Information
```
GET /model/info
```
Returns information about the loaded model.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 100,
  "n_features": 18,
  "classes": ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"]
}
```

### 4. Single Prediction
```
POST /predict
```
Predict drug for a single patient.

**Request Body:**
```json
{
  "Age": 45,
  "Sex": "M",
  "BP": "HIGH",
  "Cholesterol": "NORMAL",
  "Na_to_K": 15.5
}
```

**Valid Values:**
- `Age`: 0-100 (integer)
- `Sex`: "M" or "F"
- `BP`: "HIGH", "NORMAL", or "LOW"
- `Cholesterol`: "HIGH" or "NORMAL"
- `Na_to_K`: positive float

**Response:**
```json
{
  "prediction": "DrugY",
  "patient_data": {
    "Age": 45,
    "Sex": "M",
    "BP": "HIGH",
    "Cholesterol": "NORMAL",
    "Na_to_K": 15.5
  },
  "status": "success"
}
```

### 5. Batch Prediction
```
POST /predict/batch
```
Predict drugs for multiple patients.

**Request Body:**
```json
[
  {
    "Age": 45,
    "Sex": "M",
    "BP": "HIGH",
    "Cholesterol": "NORMAL",
    "Na_to_K": 15.5
  },
  {
    "Age": 30,
    "Sex": "F",
    "BP": "NORMAL",
    "Cholesterol": "HIGH",
    "Na_to_K": 12.0
  }
]
```

## Testing the API

### Using the test script:
```bash
# Make sure the server is running first (in another terminal)
make serve

# Then run the tests
make test-api
```

### Using curl:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Sex": "M",
    "BP": "HIGH",
    "Cholesterol": "NORMAL",
    "Na_to_K": 15.5
  }'
```

### Using Python:
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Age": 45,
        "Sex": "M",
        "BP": "HIGH",
        "Cholesterol": "NORMAL",
        "Na_to_K": 15.5
    }
)
print(response.json())
```

## Example Client

See `test_api.py` for a complete example of how to interact with all API endpoints.

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `422`: Validation Error (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Development

### Project Structure
```
.
├── app.py                 # FastAPI application
├── model_pipeline.py      # ML pipeline functions
├── main.py                # Training script
├── test_api.py           # API test script
├── requirements.txt       # Dependencies
├── Makefile              # Build commands
└── models/
    └── random_forest_model.pkl  # Trained model
```

### Security Scan
```bash
make security
```

### Code Linting
```bash
make lint
```

## Troubleshooting

### Model Not Found Error
If you get a "Model not found" error, train the model first:
```bash
make pipeline
```

### Port Already in Use
If port 8000 is already in use, change the port in `app.py` or use:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Import Errors
Make sure all dependencies are installed:
```bash
make install
```

## Production Deployment

For production deployment, consider:

1. **Use Gunicorn with Uvicorn workers:**
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. **Add authentication and rate limiting**

3. **Use HTTPS with a reverse proxy (nginx)**

4. **Monitor with logging and metrics**

5. **Containerize with Docker**

## License

This project is for educational purposes.
