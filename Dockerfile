FROM python:3.9-slim

WORKDIR /app

# Copier uniquement les dépendances d'abord pour profiter du cache Docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans l'image
COPY . .

# Exposer le port de l'API FastAPI
EXPOSE 8000

# Lancer le web service FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
