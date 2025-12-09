# 🎓 MLOps Learning Guide for Beginners

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [What is MLOps?](#what-is-mlops)
3. [File-by-File Explanation](#file-by-file-explanation)
4. [MLOps Concepts in This Project](#mlops-concepts)
5. [Learning Path](#learning-path)
6. [Hands-On Exercises](#hands-on-exercises)

---

## 🎯 Project Overview

**Project Name**: Drug Classification ML Pipeline & API

**What it does**: This project predicts which medication (DrugA, DrugB, DrugC, DrugX, or DrugY) a patient should receive based on their characteristics:
- Age
- Sex (M/F)
- Blood Pressure (HIGH/NORMAL/LOW)
- Cholesterol (HIGH/NORMAL)
- Na_to_K ratio (blood sodium to potassium ratio)

**The Big Picture**: This is a complete MLOps project that takes you from raw data to a production-ready API that can be deployed anywhere!

---

## 🤔 What is MLOps?

**MLOps = Machine Learning + Operations**

Think of it like this:
- **Machine Learning (ML)**: Building models that can make predictions
- **Operations (Ops)**: Making those models work in the real world (deploying, monitoring, updating)

### Why MLOps Matters:
1. **Scientists build models** → But models need to serve users
2. **Models need updates** → Data changes, models must adapt
3. **Production is different** → What works on your laptop must work on servers
4. **Automation saves time** → No manual repetition

---

## 📁 File-by-File Explanation

### 🔵 **Core Data & Configuration Files**

#### 1. `drug200.csv`
**What it is**: The dataset with 200 patient records
**What you learn**:
- Real-world data structure
- Features (inputs) vs target (output)
- Data for training ML models

**Contents**:
```csv
Age,Sex,BP,Cholesterol,Na_to_K,Drug
23,F,HIGH,HIGH,25.355,DrugY
47,M,LOW,HIGH,13.093,drugC
...
```

**MLOps Concept**: **Data Management** - Versioning and storing training data

---

#### 2. `requirements.txt`
**What it is**: List of all Python libraries needed

**Content breakdown**:
```txt
numpy              # Math operations
pandas             # Data manipulation
scikit-learn       # Machine learning algorithms
matplotlib         # Visualizations
jupyter            # Interactive notebooks
mlflow             # ML experiment tracking
fastapi            # Modern web API framework
uvicorn            # ASGI server for FastAPI
joblib             # Model serialization
imbalanced-learn   # Handle imbalanced datasets (SMOTE)
seaborn            # Advanced visualizations
requests           # HTTP client
flask              # Web framework for UI
```

**MLOps Concept**: **Dependency Management** - Everyone uses the same versions

**How to use**:
```bash
pip install -r requirements.txt
```

---

### 🔵 **Machine Learning Pipeline Files**

#### 3. `model_pipeline.py` ⭐ **MOST IMPORTANT FOR ML**
**What it is**: The brain of the ML operations - modular functions for every ML step

**Functions explained**:

##### `load_data(filepath)`
```python
# Loads CSV data into a pandas DataFrame
df = load_data('drug200.csv')
```
**Learns**: Data loading, error handling

---

##### `explore_data(df)`
```python
# Shows you what's in the data
# - Shape (rows, columns)
# - Data types
# - Missing values
# - Statistics
```
**Learns**: Exploratory Data Analysis (EDA)

---

##### `prepare_data(df, test_size=0.3, apply_smote=True)`
**This is CRITICAL - prepares raw data for ML**

**Steps**:
1. **Age Binning**: Groups ages into categories
   ```python
   # Instead of Age=45, we get '40s'
   # Makes patterns easier to learn
   ```

2. **Na_to_K Binning**: Groups sodium/potassium ratios
   ```python
   # Instead of 15.5, we get '10-20' range
   ```

3. **One-Hot Encoding**: Converts categories to numbers
   ```python
   # Sex='M' becomes Sex_M=1, Sex_F=0
   # ML models need numbers, not text!
   ```

4. **Train-Test Split**: Divides data
   - 70% for training (teaching the model)
   - 30% for testing (checking if it learned)

5. **SMOTE** (Synthetic Minority Over-sampling Technique)
   ```python
   # If DrugA has 100 examples but DrugB has 10
   # SMOTE creates synthetic DrugB examples
   # Prevents model from being biased
   ```

**MLOps Concept**: **Feature Engineering** & **Data Preprocessing**

---

##### `train_model(X_train, y_train, n_estimators=100, max_leaf_nodes=30)`
**Trains a Random Forest classifier**

**What is Random Forest?**
- Think of it as asking 100 expert doctors (trees)
- Each doctor votes on which drug to prescribe
- Majority vote wins!
- Very powerful and accurate

**Parameters**:
- `n_estimators=100`: Number of "doctor" trees
- `max_leaf_nodes=30`: How complex each tree can be

**MLOps Concept**: **Model Training** & **Hyperparameter Tuning**

---

##### `evaluate_model(model, X_test, y_test)`
**Tests how good the model is**

**Metrics**:
- **Accuracy**: How often is it correct? (e.g., 95%)
- **Confusion Matrix**: Which drugs does it confuse?
- **Classification Report**: Precision, recall, F1-score

**MLOps Concept**: **Model Evaluation** & **Performance Metrics**

---

##### `save_model(model, filepath)` & `load_model(filepath)`
**Saves/loads trained models using joblib**

```python
# Save once, use forever!
save_model(model, 'models/random_forest_model.pkl')

# Load anywhere
model = load_model('models/random_forest_model.pkl')
```

**MLOps Concept**: **Model Serialization** & **Model Registry**

---

##### `predict_new_data(model, X_new)`
**Makes predictions on new patients**

**MLOps Concept**: **Inference/Prediction**

---

#### 4. `main.py`
**What it is**: Command-line interface (CLI) to run the pipeline

**How it works**:
```bash
# Run everything
python main.py --action full_pipeline --data drug200.csv

# Just train
python main.py --action train --n_estimators 150

# Just evaluate
python main.py --action evaluate

# Optimize hyperparameters
python main.py --action optimize
```

**Why it's useful**:
- Run experiments from command line
- Easy to automate
- Can be called by other scripts

**MLOps Concept**: **Pipeline Orchestration** & **CLI Tools**

---

### 🔵 **API & Deployment Files**

#### 5. `app.py` ⭐ **MOST IMPORTANT FOR DEPLOYMENT**
**What it is**: FastAPI REST API that serves the ML model

**Key Components**:

##### Pydantic Models (Data Validation)
```python
class PatientData(BaseModel):
    Age: int        # Must be integer
    Sex: str        # Must be string
    BP: str         # Must be string
    Cholesterol: str
    Na_to_K: float  # Must be decimal
```
**Purpose**: Automatically validates incoming data. If someone sends bad data, it's rejected automatically!

---

##### API Endpoints (Routes)

**1. `GET /health`** - Check if API is working
```python
# Returns:
{
  "status": "healthy",
  "model_loaded": true
}
```

**2. `POST /predict`** - Make a prediction
```python
# Send:
{
  "Age": 45,
  "Sex": "M",
  "BP": "HIGH",
  "Cholesterol": "NORMAL",
  "Na_to_K": 15.5
}

# Get back:
{
  "prediction": "DrugY",
  "status": "success"
}
```

**3. `POST /retrain`** - Retrain model with new parameters
```python
# Send:
{
  "n_estimators": 150,
  "max_leaf_nodes": 40,
  "test_size": 0.3,
  "apply_smote": true
}

# Get back:
{
  "accuracy": "98.33%",
  "status": "success"
}
```

**4. `GET /model/info`** - Get model details
```python
# Returns:
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 100,
  "classes": ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"]
}
```

---

##### Key Functions

**`load_model_and_features()`**
- Loads saved model at startup
- Extracts feature columns
- **Critical**: Ensures API is ready immediately

**`preprocess_input(patient_data)`**
- Applies same transformations as training
- Age binning, Na_to_K binning, one-hot encoding
- **Critical**: Prediction data must match training data format!

**`@app.on_event("startup")`**
- Runs when API starts
- Loads model before accepting requests

---

**MLOps Concepts**: 
- **Model Serving** - Making model accessible via HTTP
- **REST API** - Standard way to expose services
- **Containerization Ready** - Can be put in Docker
- **Hot Reload** - Changes reflected without restart

---

#### 6. `flask_app.py`
**What it is**: User-friendly web interface that talks to the FastAPI backend

**Architecture**:
```
User Browser → Flask (Port 5000) → FastAPI (Port 8000) → ML Model
     ↑                                      ↓
     └──────────── Result ──────────────────┘
```

**Pages**:
1. **Home** (`/`) - Overview
2. **Predict** (`/predict`) - Form to input patient data
3. **Retrain** (`/retrain`) - Form to retrain model
4. **Model Info** (`/model-info`) - Display model details
5. **About** (`/about`) - Documentation

**Why separate Flask + FastAPI?**
- **Flask**: Easy HTML forms, user-friendly
- **FastAPI**: High-performance, auto-documentation, for ML
- **Best of both worlds!**

**MLOps Concept**: **User Interface** & **Service Integration**

---

#### 7. `test_api.py`
**What it is**: Automated tests for API endpoints

**Tests**:
```python
test_health()      # Is API running?
test_predict()     # Can it predict?
test_retrain()     # Can it retrain?
test_model_info()  # Can it return info?
```

**Why testing matters**:
- Catches bugs early
- Ensures changes don't break things
- Confidence in deployments

**MLOps Concept**: **Continuous Testing** & **API Testing**

---

### 🔵 **DevOps & Automation Files**

#### 8. `Makefile` ⭐ **AUTOMATION POWERHOUSE**
**What it is**: Automation commands for common tasks

**Key commands**:

```bash
make install     # Install all dependencies
make pipeline    # Train the model
make serve       # Start FastAPI server
make flask       # Start Flask UI
make test-api    # Run API tests
make docker-build # Create Docker container
make docker-run  # Deploy in Docker
make clean       # Remove temporary files
```

**Why it's powerful**:
- One command instead of many
- Consistent across team members
- Easy for beginners
- Professional workflow

**MLOps Concept**: **Build Automation** & **Task Orchestration**

---

#### 9. `Dockerfile`
**What it is**: Recipe to create a container with everything needed

**What is a container?**
Think of it as a complete package:
- Operating system (Linux)
- Python
- All libraries
- Your code
- Your model

**Benefits**:
- **Portable**: Runs anywhere (your laptop, cloud, server)
- **Consistent**: Same environment everywhere
- **Isolated**: Won't interfere with other apps

**Dockerfile breakdown**:
```dockerfile
FROM python:3.9-slim          # Start with Python
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy dependency list
RUN pip install -r requirements.txt  # Install dependencies
COPY . .                       # Copy all code
EXPOSE 8000                    # Open port 8000
CMD ["uvicorn", "app:app", ...] # Start the API
```

**MLOps Concept**: **Containerization** - Core DevOps practice

---

#### 10. `docker-compose.yml`
**What it is**: Orchestrates multiple containers (in this case, just the API)

**Benefits**:
- Start/stop with one command
- Configure volumes (persistent storage)
- Set environment variables
- Easy scaling

```bash
docker-compose up    # Start everything
docker-compose down  # Stop everything
```

**MLOps Concept**: **Container Orchestration**

---

### 🔵 **Documentation Files**

#### 11. `API_README.md`
- Complete API documentation
- How to use each endpoint
- Example requests/responses

#### 12. `FLASK_README.md`
- Flask interface guide (in French)
- How to use the web UI

#### 13. `ATELIER4_README.md`
- Workshop/tutorial guide (in French)
- Step-by-step learning

#### 14. `QUICKSTART.md`
- Quick setup guide
- Get running in 5 minutes

#### 15. `PRESENTATION.md`
- Project presentation
- High-level overview

**MLOps Concept**: **Documentation** - Critical for team collaboration

---

### 🔵 **Other Supporting Files**

#### 16. `example_client.py`
**What it is**: Shows how to use the API from Python

**Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"Age": 45, "Sex": "M", ...}
)
print(response.json())
```

---

#### 17. `test_pipeline.py`
**What it is**: Tests the ML pipeline functions

---

#### 18. `test_environment.py`
**What it is**: Checks if environment is set up correctly

---

#### 19. `watch.py`
**What it is**: Auto-reruns pipeline when files change

**Use case**: During development, automatically retrain when you modify code

---

#### 20. `install.sh`
**What it is**: Shell script to automate setup

---

### 📂 **Directories**

#### `models/`
- Stores trained models (`.pkl` files)
- Model registry

#### `templates/`
- HTML templates for Flask
- `base.html` - Common layout
- `index.html` - Home page
- `predict.html` - Prediction form
- `retrain.html` - Retrain form
- `model_info.html` - Model info display
- `about.html` - About page

#### `static/`
- CSS, JavaScript, images (currently using CDN)

#### `__pycache__/`
- Python bytecode cache (auto-generated, ignore it)

---

## 🎯 MLOps Concepts in This Project

### 1. **Version Control** (Git)
Your project likely uses Git. Learn:
```bash
git add .
git commit -m "Added feature"
git push
```

### 2. **Virtual Environments**
Isolate project dependencies:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. **Data Versioning**
Track data changes (this project uses CSV; advanced: DVC)

### 4. **Model Versioning**
Save models with timestamps:
```python
model_v1.pkl
model_v2.pkl
```

### 5. **Continuous Integration/Continuous Deployment (CI/CD)**
Automate testing and deployment (advanced topic)

### 6. **Monitoring & Logging**
Track model performance in production (add later)

### 7. **A/B Testing**
Compare model versions (advanced)

### 8. **Model Registry**
Centralized model storage (MLflow, advanced)

---

## 🚀 Learning Path

### **Week 1: Understand the ML Pipeline**
1. Read `model_pipeline.py` line by line
2. Run: `make pipeline`
3. Observe each step's output
4. Try changing hyperparameters

**Exercises**:
- Change `n_estimators` from 100 to 50
- Disable SMOTE, see accuracy change
- Modify age bins

---

### **Week 2: Master the API**
1. Read `app.py` carefully
2. Start API: `make serve`
3. Open Swagger: http://localhost:8000/docs
4. Test each endpoint manually
5. Run automated tests: `make test-api`

**Exercises**:
- Add a new endpoint `/hello`
- Modify prediction response format
- Add input validation (age must be 0-100)

---

### **Week 3: Build the UI**
1. Read `flask_app.py`
2. Start Flask: `make flask`
3. Test all pages
4. Understand Flask → FastAPI communication

**Exercises**:
- Add a "History" page showing past predictions
- Style the UI with custom CSS
- Add a chart showing model accuracy

---

### **Week 4: Docker & Deployment**
1. Read `Dockerfile` and `docker-compose.yml`
2. Build image: `make docker-build`
3. Run container: `make docker-run`
4. Test API in Docker

**Exercises**:
- Deploy to a cloud provider (Heroku, AWS, Google Cloud)
- Add HTTPS/SSL
- Set up CI/CD with GitHub Actions

---

## 🧪 Hands-On Exercises

### Beginner Level

#### Exercise 1: Change Hyperparameters
```bash
# Train with different parameters
python main.py --action full_pipeline --n_estimators 150 --max_leaf_nodes 40
```
**Goal**: Understand how parameters affect accuracy

---

#### Exercise 2: Test Different Patients
**Using Swagger UI**:
1. Go to http://localhost:8000/docs
2. Try predicting for these patients:

**Patient A** (High risk):
```json
{
  "Age": 70,
  "Sex": "M",
  "BP": "HIGH",
  "Cholesterol": "HIGH",
  "Na_to_K": 25.0
}
```

**Patient B** (Low risk):
```json
{
  "Age": 25,
  "Sex": "F",
  "BP": "NORMAL",
  "Cholesterol": "NORMAL",
  "Na_to_K": 12.0
}
```

**Goal**: Understand model behavior

---

#### Exercise 3: Add Logging
Modify `app.py` to log predictions:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(patient_data: PatientData):
    # ... existing code ...
    logger.info(f"Prediction made: {predicted_drug} for patient age {patient_data.Age}")
    # ... rest of code ...
```

**Goal**: Learn about production monitoring

---

### Intermediate Level

#### Exercise 4: Add a New Feature
Add "Weight" to the model:
1. Modify dataset (add Weight column)
2. Update `prepare_data()` to handle Weight
3. Update `PatientData` schema in `app.py`
4. Retrain model
5. Test predictions

**Goal**: Full-stack ML feature development

---

#### Exercise 5: Implement Model Versioning
```python
# In app.py, save models with timestamps
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/rf_model_{timestamp}.pkl"
save_model(model, model_path)
```

**Goal**: Track model evolution

---

#### Exercise 6: Add Database Storage
Use SQLite to store predictions:
```python
import sqlite3

conn = sqlite3.connect('predictions.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (timestamp TEXT, age INT, sex TEXT, prediction TEXT)''')

# After prediction
c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
          (datetime.now(), age, sex, prediction))
conn.commit()
```

**Goal**: Persist prediction history

---

### Advanced Level

#### Exercise 7: Implement A/B Testing
Run two models simultaneously, compare results:
```python
model_a = load_model('models/model_v1.pkl')
model_b = load_model('models/model_v2.pkl')

prediction_a = model_a.predict(X)
prediction_b = model_b.predict(X)

# Return both, track which performs better
```

---

#### Exercise 8: Add Monitoring Dashboard
Use Streamlit or Grafana to visualize:
- Predictions per hour
- Model accuracy over time
- Most common predictions

---

#### Exercise 9: Implement Model Explainability
Add SHAP or LIME to explain predictions:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Return feature importance with prediction
```

---

## 📚 Key Takeaways

### What You've Learned

1. **Complete ML Pipeline**: Data → Model → Deployment
2. **API Design**: RESTful services with FastAPI
3. **Containerization**: Docker for deployment
4. **Automation**: Makefiles for efficiency
5. **Testing**: Automated API testing
6. **Documentation**: Professional documentation practices

### Next Steps to Become MLOps Engineer

1. **Learn Git Advanced**: Branching, merging, pull requests
2. **Master Docker**: Multi-stage builds, docker-compose
3. **CI/CD**: GitHub Actions, Jenkins, GitLab CI
4. **Cloud Platforms**: AWS SageMaker, Google Cloud AI Platform, Azure ML
5. **Kubernetes**: Container orchestration at scale
6. **MLflow/Kubeflow**: ML experiment tracking and pipelines
7. **Monitoring**: Prometheus, Grafana, ELK stack
8. **Model Serving**: TensorFlow Serving, Seldon, KServe

### Resources

**Books**:
- "Introducing MLOps" by Mark Treveil
- "Building Machine Learning Pipelines" by Hannes Hapke

**Courses**:
- Coursera: "Machine Learning Engineering for Production (MLOps)"
- Udemy: "MLOps Fundamentals"

**Practice Platforms**:
- Kaggle (ML competitions)
- GitHub (open-source MLOps projects)

---

## 🎉 Conclusion

This project is a **complete, production-ready MLOps system**. You have:

✅ Data pipeline with preprocessing  
✅ Model training with hyperparameter tuning  
✅ REST API for serving predictions  
✅ Web interface for users  
✅ Automated testing  
✅ Docker containerization  
✅ Documentation  
✅ Automation with Make  

**This is enterprise-grade MLOps!** Study it deeply, experiment with it, break it, fix it, and you'll gain invaluable real-world experience.

Keep learning, keep building! 🚀

---

## 📞 Getting Help

**When stuck**:
1. Read error messages carefully
2. Check the documentation (README files)
3. Use Swagger UI to debug API issues
4. Print/log variables to understand flow
5. Google the specific error
6. Ask on Stack Overflow or Reddit (r/MachineLearning)

**Remember**: Every expert was once a beginner. Keep coding! 💪
