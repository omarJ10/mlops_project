# 🚀 Quick Start Guide - Drug Classification API

This guide will help you quickly deploy your ML model as a FastAPI service.

## Prerequisites

- Python 3.8+
- Virtual environment (`venv`)
- Trained model (or run the training pipeline)

## 1. Initial Setup

```bash
# Navigate to project directory
cd ~/omar-jalled-4ds8-ml_project

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
make install
```

## 2. Train the Model (if not already trained)

```bash
# Run the complete ML pipeline
make pipeline
```

This will:
- Load and explore the data
- Prepare features (binning, encoding, SMOTE)
- Train a Random Forest model
- Evaluate performance
- Save the model to `models/random_forest_model.pkl`

## 3. Start the FastAPI Server

### Option A: Using Make (Recommended)
```bash
make serve
```

### Option B: Using Python directly
```bash
python app.py
```

### Option C: Using uvicorn
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 4. Test the API

### In a new terminal:

```bash
# Activate venv
source venv/bin/activate

# Run API tests
make test-api
```

### Or test manually with curl:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
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

## 5. Use the Example Client

```bash
# Run the example client
python example_client.py
```

This demonstrates:
- Single predictions
- Batch predictions
- Different patient scenarios

## 🐳 Docker Deployment (Optional)

### Build and run with Docker:

```bash
# Build the Docker image
make docker-build

# Run the container
make docker-run

# View logs
docker-compose logs -f

# Stop the container
make docker-stop
```

## 📊 API Usage Examples

### Python Client

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

result = response.json()
print(f"Predicted Drug: {result['prediction']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function predictDrug() {
  const response = await axios.post('http://localhost:8000/predict', {
    Age: 45,
    Sex: 'M',
    BP: 'HIGH',
    Cholesterol: 'NORMAL',
    Na_to_K: 15.5
  });
  
  console.log(`Predicted Drug: ${response.data.prediction}`);
}

predictDrug();
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Sex": "M",
    "BP": "HIGH",
    "Cholesterol": "NORMAL",
    "Na_to_K": 15.5
  }'
```

## 📝 Available Make Commands

```bash
make help          # Show all available commands
make install       # Install Python dependencies
make pipeline      # Train the ML model
make serve         # Start the FastAPI server
make test-api      # Test API endpoints
make lint          # Run code linting
make security      # Run security checks
make clean         # Clean temporary files

# Docker commands
make docker-build  # Build Docker image
make docker-run    # Run in Docker container
make docker-stop   # Stop Docker container
```

## 🔍 Troubleshooting

### Model Not Found
```bash
# Train the model first
make pipeline
```

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app:app --port 8080
```

### Import Errors
```bash
# Reinstall dependencies
make install
```

### Permission Errors
```bash
# Make scripts executable
chmod +x install.sh watch.py
```

## 🎯 Next Steps

1. **Explore Interactive Docs**: Visit http://localhost:8000/docs to try all endpoints
2. **Integrate with Your App**: Use the Python/JavaScript examples above
3. **Deploy to Production**: Consider using Docker, AWS, Azure, or GCP
4. **Add Authentication**: Implement API keys or OAuth
5. **Monitor Performance**: Add logging and metrics

## 📖 Full Documentation

See `API_README.md` for complete API documentation.

## 💡 Tips

- Use `/docs` for interactive API testing
- Check `/health` to verify model is loaded
- Use `/model/info` to see model details
- Batch predictions are more efficient for multiple patients
- Enable auto-reload during development (`--reload` flag)

## 🚀 Production Deployment

For production, consider:

1. Use Gunicorn with Uvicorn workers
2. Add HTTPS/SSL certificates
3. Implement rate limiting
4. Add authentication (API keys, OAuth)
5. Use Docker for containerization
6. Deploy to cloud (AWS, Azure, GCP)
7. Add monitoring and logging
8. Use a reverse proxy (nginx)

---

**Need help?** Check the full documentation in `API_README.md` or open an issue.
