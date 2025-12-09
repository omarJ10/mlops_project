# 📋 Quick Reference Cheat Sheet

## Essential Commands

```bash
# Setup
source venv/bin/activate    # Activate environment
make install                # Install dependencies

# Training
make pipeline               # Train model (default settings)
python main.py --action full_pipeline --n_estimators 150  # Custom

# Serving
make serve                  # Start API (port 8000)
make flask                  # Start web UI (port 5000)

# Testing
make test-api               # Test API endpoints
curl http://localhost:8000/health  # Quick health check

# Docker
make docker-build           # Build image
make docker-run             # Run container
make docker-stop            # Stop container

# Cleanup
make clean                  # Remove temp files
```

---

## File Roles at a Glance

| File | Purpose | When to Edit |
|------|---------|--------------|
| `drug200.csv` | Training data | Never (or add more data) |
| `model_pipeline.py` | ML logic | Add features, algorithms |
| `main.py` | CLI interface | Add commands |
| `app.py` | FastAPI server | Add endpoints, modify API |
| `flask_app.py` | Web UI | Change interface |
| `requirements.txt` | Dependencies | Add new libraries |
| `Dockerfile` | Container recipe | Change deployment setup |
| `Makefile` | Automation | Add new commands |
| `test_api.py` | API tests | Add test cases |

---

## API Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/` | GET | API info | `curl http://localhost:8000/` |
| `/health` | GET | Check status | `curl http://localhost:8000/health` |
| `/predict` | POST | Single prediction | See below |
| `/retrain` | POST | Retrain model | See below |
| `/model/info` | GET | Model details | `curl http://localhost:8000/model/info` |
| `/docs` | GET | Swagger UI | Open in browser |

### Prediction Example
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

### Retrain Example
```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "n_estimators": 150,
    "max_leaf_nodes": 40,
    "test_size": 0.3,
    "apply_smote": true
  }'
```

---

## Python Quick Reference

### Load and Use Model
```python
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Make prediction
prediction = model.predict(X_new)
```

### Train New Model
```python
from model_pipeline import load_data, prepare_data, train_model

# Load and prepare data
df = load_data('drug200.csv')
X_train, X_test, y_train, y_test = prepare_data(df)

# Train
model = train_model(X_train, y_train, n_estimators=100)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")
```

### Call API from Python
```python
import requests

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
print(f"Predicted: {result['prediction']}")
```

---

## Data Preprocessing Steps

1. **Age Binning**: 45 → '40s'
2. **Na_to_K Binning**: 15.5 → '10-20'
3. **One-Hot Encoding**: 'M' → Sex_M=1, Sex_F=0
4. **Train-Test Split**: 70% train, 30% test
5. **SMOTE** (optional): Balance classes

---

## Model Parameters

### Random Forest
- `n_estimators`: Number of trees (default: 100)
  - More trees = better accuracy, slower training
  - Try: 50, 100, 150, 200

- `max_leaf_nodes`: Tree complexity (default: 30)
  - Higher = more complex trees
  - Try: 10, 30, 50, 100

- `random_state`: For reproducibility (default: 1)
  - Same seed = same results

### Training Options
- `test_size`: Test proportion (default: 0.3)
  - 0.3 = 30% test, 70% train
  
- `apply_smote`: Balance classes (default: True)
  - True = balanced classes
  - False = natural class distribution

---

## Docker Commands

```bash
# Build image
docker build -t drug-api .

# Run container
docker run -p 8000:8000 drug-api

# Run in background
docker run -d -p 8000:8000 drug-api

# List running containers
docker ps

# Stop container
docker stop <container-id>

# View logs
docker logs <container-id>
docker logs -f <container-id>  # Follow logs

# Enter container
docker exec -it <container-id> bash

# Remove container
docker rm <container-id>

# Remove image
docker rmi drug-api

# With docker-compose
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose logs -f    # View logs
```

---

## Troubleshooting

### Model Not Found
```bash
# Solution: Train the model first
make pipeline
```

### Port Already in Use
```bash
# Solution 1: Kill process
lsof -ti:8000 | xargs kill -9

# Solution 2: Use different port
uvicorn app:app --port 8080
```

### Import Errors
```bash
# Solution: Reinstall dependencies
make install
```

### Virtual Environment Not Activated
```bash
# Activate it
source venv/bin/activate

# Check
which python  # Should point to venv/bin/python
```

### API Not Responding
```bash
# Check if running
curl http://localhost:8000/health

# Check logs
# (Look at terminal where you ran 'make serve')

# Restart
# Ctrl+C to stop, then 'make serve' again
```

### Flask Can't Connect to API
```bash
# Make sure API is running
curl http://localhost:8000/health

# Check API_URL in flask_app.py
# Should be: API_URL = "http://localhost:8000"
```

---

## Git Commands (if using version control)

```bash
# Initialize repo
git init

# Add files
git add .

# Commit
git commit -m "Initial commit"

# Create .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.png" >> .gitignore

# Push to GitHub
git remote add origin <your-repo-url>
git push -u origin main
```

---

## URLs to Remember

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | API base |
| Swagger UI | http://localhost:8000/docs | Interactive API docs |
| ReDoc | http://localhost:8000/redoc | Alternative docs |
| Flask UI | http://localhost:5000 | Web interface |

---

## Common Workflows

### Workflow 1: Train and Deploy
```bash
# 1. Train model
make pipeline

# 2. Start API
make serve

# 3. Test
make test-api

# 4. Use web interface
make flask  # In another terminal
```

### Workflow 2: Modify and Test
```bash
# 1. Edit code
nano model_pipeline.py

# 2. Retrain
make pipeline

# 3. Restart API
# Ctrl+C in API terminal
make serve

# 4. Test changes
curl http://localhost:8000/predict -X POST -H "Content-Type: application/json" -d '...'
```

### Workflow 3: Docker Deployment
```bash
# 1. Ensure model trained
make pipeline

# 2. Build image
make docker-build

# 3. Run container
make docker-run

# 4. Test
curl http://localhost:8000/health

# 5. Stop when done
make docker-stop
```

---

## Performance Tips

### Speed Up Training
```bash
# Use fewer trees
python main.py --action full_pipeline --n_estimators 50

# Disable SMOTE
python main.py --action full_pipeline --no_smote
```

### Improve Accuracy
```bash
# Use more trees
python main.py --action full_pipeline --n_estimators 200

# Optimize hyperparameters
python main.py --action optimize
```

### Faster API Response
- Keep model loaded (already done in app.py)
- Use smaller models
- Cache predictions for same inputs
- Use async/await for I/O operations

---

## Pydantic Validation

### Current Validation
```python
class PatientData(BaseModel):
    Age: int           # Must be integer
    Sex: str           # Must be string
    BP: str            # Must be string
    Cholesterol: str   # Must be string
    Na_to_K: float     # Must be float
```

### Enhanced Validation (add this)
```python
from pydantic import BaseModel, Field

class PatientData(BaseModel):
    Age: int = Field(..., ge=0, le=120)  # 0 to 120
    Sex: str = Field(..., regex="^[MF]$")  # Only M or F
    BP: str = Field(..., regex="^(HIGH|NORMAL|LOW)$")
    Cholesterol: str = Field(..., regex="^(HIGH|NORMAL)$")
    Na_to_K: float = Field(..., gt=0)  # Positive only
```

---

## Environment Variables

### Set in Terminal
```bash
export MODEL_PATH="models/random_forest_model.pkl"
export API_PORT=8000
export DEBUG=True
```

### Use in Python
```python
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_model.pkl")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False") == "True"
```

### Set in Docker
```dockerfile
ENV MODEL_PATH="/app/models/random_forest_model.pkl"
ENV API_PORT=8000
```

---

## Logging Levels

```python
import logging

# Set level
logging.basicConfig(level=logging.DEBUG)  # Most verbose
logging.basicConfig(level=logging.INFO)   # Standard
logging.basicConfig(level=logging.WARNING)  # Only warnings+
logging.basicConfig(level=logging.ERROR)   # Only errors

# Use logger
logger = logging.getLogger(__name__)
logger.debug("Detailed debug info")
logger.info("General info")
logger.warning("Warning message")
logger.error("Error occurred")
```

---

## Model Metrics Explained

### Accuracy
- **What**: Percentage of correct predictions
- **Formula**: (Correct Predictions) / (Total Predictions)
- **Example**: 95% means 95 out of 100 correct

### Confusion Matrix
```
                Predicted
              A    B    C
Actual    A   10   0    1   ← A: 10 correct, 1 wrong
          B   0    15   0   ← B: all correct
          C   0    1    12  ← C: 12 correct, 1 wrong
```

### Precision
- **What**: Of all predicted as X, how many are truly X?
- **Formula**: True Positives / (True Positives + False Positives)

### Recall
- **What**: Of all true X, how many did we predict as X?
- **Formula**: True Positives / (True Positives + False Negatives)

### F1-Score
- **What**: Harmonic mean of precision and recall
- **Range**: 0 to 1 (higher is better)

---

## MLOps Checklist

When building an ML project, ensure you have:

- [ ] **Data Management**: Version your datasets
- [ ] **Code Version Control**: Use Git
- [ ] **Environment Management**: Requirements.txt or Conda
- [ ] **Model Versioning**: Save models with timestamps
- [ ] **Testing**: Unit tests, integration tests
- [ ] **API Documentation**: Swagger/OpenAPI
- [ ] **Containerization**: Dockerfile
- [ ] **CI/CD**: Automated testing and deployment
- [ ] **Monitoring**: Log predictions and errors
- [ ] **Documentation**: README, guides, comments
- [ ] **Error Handling**: Graceful failure
- [ ] **Security**: Input validation, authentication

---

## Common Mistakes to Avoid

1. **Not activating venv**: Always run `source venv/bin/activate`
2. **Training without saving**: Always save your model
3. **Preprocessing mismatch**: Same preprocessing for train and predict
4. **No error handling**: Always use try-except
5. **Hardcoded values**: Use config files or env variables
6. **No validation**: Validate all inputs
7. **No logging**: Add logs for debugging
8. **No tests**: Write tests early
9. **No documentation**: Document as you code
10. **No version control**: Use Git from day 1

---

## Quick Debugging

### Check Python Path
```bash
which python     # Should be in venv
python --version # Check version
```

### Check Imports
```bash
python -c "import fastapi; print('FastAPI OK')"
python -c "import sklearn; print('Scikit-learn OK')"
```

### Check Model File
```bash
ls -lh models/random_forest_model.pkl
# Should exist and be > 0 bytes
```

### Check API is Running
```bash
ps aux | grep uvicorn
# Should see running process

netstat -tuln | grep 8000
# Should see port 8000 LISTEN
```

### Check Logs
```bash
# API logs (in terminal running 'make serve')
# Flask logs (in terminal running 'make flask')

# Docker logs
docker-compose logs -f
```

---

## Useful Python Snippets

### Load Data and Explore
```python
import pandas as pd

df = pd.read_csv('drug200.csv')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df['Drug'].value_counts())
```

### Quick Prediction
```python
import joblib
import pandas as pd

model = joblib.load('models/random_forest_model.pkl')

# Create sample (must match training preprocessing!)
# This is simplified - use preprocess_input() for real predictions
```

### Batch Predictions
```python
# Read multiple patients from CSV
patients = pd.read_csv('new_patients.csv')

# Preprocess
from app import preprocess_input
# Process each patient...

# Predict all at once
predictions = model.predict(X_new)
```

---

## Resources Links

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Scikit-learn Docs**: https://scikit-learn.org/
- **Docker Docs**: https://docs.docker.com/
- **Flask Docs**: https://flask.palletsprojects.com/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **MLOps Community**: https://mlops.community/
- **Kaggle Datasets**: https://www.kaggle.com/datasets

---

## Your Project Checklist

Track your progress:

- [ ] Day 1: Setup and first run ✓
- [ ] Day 2: Understand ML pipeline ✓
- [ ] Day 3: Understand API ✓
- [ ] Day 4: Docker & deployment ✓
- [ ] Day 5: Flask integration ✓
- [ ] Day 6: Testing & quality ✓
- [ ] Day 7: Build your own project ✓

---

## Keep This Handy!

Print or bookmark this page. You'll refer to it often while learning!

**Happy coding! 🚀**
