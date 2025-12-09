# MLOps Drug Classification Project

A production-ready machine learning project for drug classification using Random Forest, with FastAPI REST API, Flask web interface, MLflow experiment tracking, and Docker deployment.

## 📁 Project Structure

```
mlops_project/
├── src/                          # Source code
│   ├── app.py                   # FastAPI REST API
│   ├── flask_app.py             # Flask web interface
│   ├── model_pipeline.py        # ML pipeline (train, predict, evaluate)
│   └── main.py                  # CLI interface for pipeline
├── tests/                        # Test files
│   ├── test_api.py              # API tests
│   └── test_pipeline.py         # Pipeline tests
├── docs/                         # Documentation
│   ├── PRESENTATION.md          # Project presentation
│   └── QUICKSTART.md            # Quick start guide
├── scripts/                      # Utility scripts
│   ├── install.sh               # Installation script
│   ├── start_mlflow.sh          # MLflow startup script
│   └── cleanup_unnecessary.sh   # Cleanup script
├── learning/                     # Educational materials
│   ├── START_HERE.md
│   ├── MLOPS_LEARNING_GUIDE.md
│   ├── QUICK_REFERENCE.md
│   └── ...
├── data/                         # Data files
│   └── drug200.csv              # Training dataset
├── models/                       # Saved models
│   └── random_forest_model.pkl
├── templates/                    # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   └── ...
├── config/                       # Configuration files
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── Makefile                     # Automation commands
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Make (optional but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/omarJ10/mlops_project.git
cd mlops_project
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
make install
# or
pip install -r requirements.txt
```

### Usage

**Train the model:**
```bash
make pipeline
```

**Start FastAPI server:**
```bash
make serve
# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Start Flask web interface:**
```bash
make flask
# Access at: http://localhost:5000
```

**Start MLflow UI:**
```bash
make mlflow
# Access at: http://localhost:5000
```

**Run tests:**
```bash
make test
```

### Docker Deployment

**Build and run with Docker:**
```bash
make docker-build
make docker-run
# API available at: http://localhost:8000
```

**Or use Docker Compose:**
```bash
docker-compose up --build
```

## 📊 Features

- ✅ Random Forest classifier for drug prediction
- ✅ FastAPI REST API with auto-generated docs
- ✅ Flask web interface for easy interaction
- ✅ MLflow experiment tracking
- ✅ Docker containerization
- ✅ Automated testing
- ✅ Code quality checks (pylint, bandit)
- ✅ Model retraining via API

## 🛠️ Available Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make pipeline      # Run ML pipeline
make serve         # Start FastAPI server
make flask         # Start Flask interface
make mlflow        # Start MLflow UI
make test          # Run tests
make lint          # Run code linting
make security      # Run security checks
make docker-build  # Build Docker image
make docker-run    # Run Docker container
make docker-stop   # Stop Docker container
```

## 📚 Documentation

- **Quick Start**: See `docs/QUICKSTART.md`
- **Full Presentation**: See `docs/PRESENTATION.md`
- **Learning Guide**: See `learning/START_HERE.md`
- **API Documentation**: http://localhost:8000/docs (when server is running)

## 🧪 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make drug prediction
- `POST /retrain` - Retrain model with new parameters
- `GET /model/info` - Get model information

## 🏗️ Technology Stack

- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI, uvicorn
- **Web**: Flask, Bootstrap 5
- **Tracking**: MLflow
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest (via test files)
- **Quality**: pylint, bandit

## 👤 Author

Omar Jalled - 4DS8

## 📄 License

This project is part of an MLOps workshop/course.
