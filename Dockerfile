FROM python:3.9-slim

WORKDIR /app

# Copier uniquement les dépendances d'abord pour profiter du cache Docker
COPY requirements.txt .

# Install with increased timeout and retry
RUN pip install --default-timeout=1000 --retries 5 -r requirements.txt

# Copier tout le projet dans l'image
COPY . .

# Exposer le port de l'API FastAPI
EXPOSE 8000

# Lancer le web service FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
