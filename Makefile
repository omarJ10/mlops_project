# Simple Makefile for Drug Classification Project

# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DATA_FILE := drug200.csv
DOCKER_IMAGE_NAME := omar_jalled_4ds8_mlops
DOCKER_TAG := latest
DOCKER_CONTAINER_NAME := omar_jalled_4ds8_mlops_container
DOCKER_HUB_REPO := omarj11/$(DOCKER_IMAGE_NAME)

# Display help with all available commands
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make pipeline   - Run complete ML pipeline"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting with pylint"
	@echo "  make security   - Run security checks with bandit"
	@echo "  make watch      - Watch files and auto-run pipeline on changes"
	@echo "  make serve      - Start FastAPI server for model deployment"
	@echo "  make flask      - Start Flask web interface"
	@echo "  make mlflow     - Start MLflow UI to view experiments and metrics"
	@echo "  make test-api   - Test the FastAPI endpoints"
	@echo "  make docker-build   - Build Docker image for API"
	@echo "  make docker-run     - Run API in Docker container"
	@echo "  make docker-stop    - Stop Docker container"
	@echo "  make docker-login   - Docker Hub login"
	@echo "  make docker-tag     - Tag image for Docker Hub"
	@echo "  make docker-push    - Push image to Docker Hub"
	@echo "  make clean      - Clean temporary files"

# Install required packages
.PHONY: install
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Done!"

# Run the complete pipeline
.PHONY: pipeline
pipeline:
	@echo "Running ML pipeline..."
	$(PYTHON) main.py --action full_pipeline --data $(DATA_FILE) --no_smote --n_estimators 100
	@echo "Pipeline completed!"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) test_pipeline.py
	@echo "Tests completed!"

# Run code linting
.PHONY: lint
lint:
	@echo "Running code linting with pylint..."
	@$(PYTHON) -m pylint main model_pipeline --disable=C0114,C0116 || true
	@echo "Linting completed!"

# Run security checks
.PHONY: security
security:
	@echo "Running security checks with bandit..."
	@$(PYTHON) -m bandit model_pipeline.py main.py -f screen || true
	@echo "Security checks completed!"

# Watch files and auto-run pipeline on changes
.PHONY: watch
watch:
	@echo "Starting file watcher..."
	@echo "The pipeline will automatically run when you modify:"
	@echo "  - model_pipeline.py"
	@echo "  - main.py"
	@echo "  - drug200.csv"
	@echo ""
	@$(PYTHON) watch.py

# Start FastAPI server (Atelier 4 - Étape 04)
.PHONY: serve
serve:
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Démarrage du serveur FastAPI"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "  API disponible sur : http://localhost:8000"
	@echo "  Documentation Swagger : http://localhost:8000/docs"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Start Flask web interface
.PHONY: flask
flask:
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Démarrage de l'interface Flask"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Interface web : http://localhost:5000"
	@echo "  Assurez-vous que l'API tourne sur le port 8000"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@$(PYTHON) flask_app.py

# Start MLflow UI
.PHONY: mlflow
mlflow:
	@./start_mlflow.sh

# Test API avec Swagger (Atelier 4 - Étape 04)
.PHONY: test-api
test-api:
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Test de l'API via Swagger UI"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "  1. Démarrez le serveur : make serve"
	@echo "  2. Ouvrez votre navigateur : http://localhost:8000/docs"
	@echo "  3. Testez les endpoints /predict et /retrain"
	@echo ""
	@echo "  Ou exécutez les tests automatiques :"
	@echo ""
	@$(PYTHON) test_api.py

# Build Docker image
.PHONY: docker-build
docker-build:
	@echo "Building Docker image for Drug Classification API..."
	@echo "Image name : $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"
	@docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .
	@echo "Docker image built successfully!"

# Run API in Docker container
.PHONY: docker-run
docker-run:
	@echo "Starting Drug Classification API in Docker..."
	@docker run -d --name $(DOCKER_CONTAINER_NAME) -p 8000:8000 $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "API is running at http://localhost:8000"
	@echo "View logs with: docker logs -f $(DOCKER_CONTAINER_NAME)"

# Stop Docker container
.PHONY: docker-stop
docker-stop:
	@echo "Stopping Docker container..."
	@docker stop $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	@echo "Container stopped and removed!"

# Docker Hub login
.PHONY: docker-login
docker-login:
	@docker login

# Tag image for Docker Hub
.PHONY: docker-tag
docker-tag:
	@echo "Tagging image for Docker Hub..."
	@echo "Local image : $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"
	@echo "Remote image: $(DOCKER_HUB_REPO):$(DOCKER_TAG)"
	@docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(DOCKER_HUB_REPO):$(DOCKER_TAG)
	@echo "Image tagged successfully!"

# Push image to Docker Hub
.PHONY: docker-push
docker-push:
	@echo "Pushing image to Docker Hub..."
	@docker push $(DOCKER_HUB_REPO):$(DOCKER_TAG)
	@echo "Image pushed successfully!"

# Clean temporary files
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -f *.png 2>/dev/null || true
	@echo "Cleanup completed!"

# Default target shows help
.DEFAULT_GOAL := help