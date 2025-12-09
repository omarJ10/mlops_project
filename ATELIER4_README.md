# Atelier 4 : Exposition de la Fonction Predict via FastAPI

## 📚 Objectif

Exposer la fonction `predict()` comme un service REST avec FastAPI.

## ✅ Livrables

- ✓ Fichier `app.py` : Service REST avec route `/predict`
- ✓ Route `/retrain` : Ré-entraînement du modèle (Excellence)
- ✓ Documentation Swagger interactive
- ✓ Tests API automatisés
- ✓ Commandes Makefile pour démarrer et tester l'API

## 🚀 Installation

### Étape 1 : Activer l'environnement virtuel

```bash
source venv/bin/activate
```

### Étape 2 : Installer FastAPI et Uvicorn

```bash
pip install fastapi uvicorn
```

Ou utilisez le Makefile :

```bash
make install
```

### Étape 3 : Entraîner le modèle (si nécessaire)

```bash
make pipeline
```

## 🌐 Démarrer l'API

### Commande Makefile (Recommandé)

```bash
make serve
```

### Commande uvicorn directe

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Commande Python

```bash
python app.py
```

## 📋 Accès à l'API

- **API Base** : http://localhost:8000
- **Documentation Swagger** : http://localhost:8000/docs ← **TESTEZ ICI !**
- **ReDoc** : http://localhost:8000/redoc

## 🧪 Tester l'API

### Option 1 : Via Swagger UI (Recommandé pour l'apprentissage)

1. Ouvrez http://localhost:8000/docs
2. Cliquez sur l'endpoint `/predict`
3. Cliquez sur "Try it out"
4. Modifiez les données patient
5. Cliquez sur "Execute"
6. Voyez le résultat de la prédiction !

### Option 2 : Via le script de test

Dans un **nouveau terminal** :

```bash
source venv/bin/activate
make test-api
```

Ou directement :

```bash
python test_api.py
```

### Option 3 : Via curl

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

## 📍 Endpoints Disponibles

### 1. GET `/` 
Page d'accueil de l'API

### 2. GET `/health`
Vérifier l'état de l'API et du modèle

**Exemple de réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/random_forest_model.pkl",
  "version": "1.0.0"
}
```

### 3. POST `/predict` ⭐
**Route principale** : Effectuer une prédiction

**Requête :**
```json
{
  "Age": 45,
  "Sex": "M",
  "BP": "HIGH",
  "Cholesterol": "NORMAL",
  "Na_to_K": 15.5
}
```

**Réponse :**
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

### 4. POST `/retrain` 🌟 (Excellence)
Ré-entraîner le modèle avec de nouveaux hyperparamètres

**Requête :**
```json
{
  "n_estimators": 100,
  "max_leaf_nodes": 30,
  "test_size": 0.3,
  "apply_smote": true
}
```

**Réponse :**
```json
{
  "status": "success",
  "message": "Modèle ré-entraîné avec succès",
  "accuracy": "98.33%",
  "parameters": {...},
  "model_path": "models/random_forest_model.pkl"
}
```

### 5. GET `/model/info`
Obtenir des informations sur le modèle chargé

## 💡 Utilisation de l'API

### Exemple Python

```python
import requests

# Prédiction
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
print(f"Médicament prédit: {result['prediction']}")
```

### Exemple JavaScript

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    Age: 45,
    Sex: 'M',
    BP: 'HIGH',
    Cholesterol: 'NORMAL',
    Na_to_K: 15.5
  })
})
.then(res => res.json())
.then(data => console.log('Médicament prédit:', data.prediction));
```

## 📝 Commandes Makefile

```bash
make install     # Installer les dépendances
make pipeline    # Entraîner le modèle
make serve       # Démarrer l'API
make test-api    # Tester l'API
make security    # Scanner la sécurité
make lint        # Vérifier le code
```

## 🔧 Structure des Fichiers

```
omar-jalled-4ds8-ml_project/
├── app.py                    # Application FastAPI (Livrable principal)
├── test_api.py              # Tests automatisés
├── model_pipeline.py         # Pipeline ML
├── main.py                   # Script d'entraînement
├── requirements.txt          # Dépendances (fastapi, uvicorn)
├── Makefile                  # Commandes automatisées
└── models/
    └── random_forest_model.pkl  # Modèle entraîné
```

## 🐛 Résolution de Problèmes

### Erreur : "Modèle non trouvé"
```bash
make pipeline  # Entraînez d'abord le modèle
```

### Erreur : "Port 8000 déjà utilisé"
```bash
# Tuez le processus sur le port 8000
lsof -ti:8000 | xargs kill -9

# Ou utilisez un autre port
uvicorn app:app --port 8080
```

### Erreur : "Module fastapi not found"
```bash
pip install fastapi uvicorn
# ou
make install
```

## 📖 Documentation FastAPI

- **Pydantic** : Validation automatique des données
- **Swagger UI** : Documentation interactive auto-générée
- **HTTPException** : Gestion des erreurs REST
- **@app.on_event("startup")** : Chargement du modèle au démarrage

## 🎯 Points Clés de l'Atelier

1. ✅ **Route `/predict`** : Exposition de la fonction predict()
2. ✅ **Chargement du modèle** : Depuis `models/random_forest_model.pkl`
3. ✅ **Documentation Swagger** : http://localhost:8000/docs
4. ✅ **Gestion des erreurs** : HTTPException avec codes appropriés
5. ✅ **Excellence : Route `/retrain`** : Ré-entraînement via REST

## 🌟 Excellence - Fonctionnalités Bonus

- ✓ Endpoint `/retrain` pour ré-entraîner le modèle
- ✓ Validation automatique avec Pydantic
- ✓ Gestion complète des erreurs
- ✓ Tests automatisés
- ✓ Documentation interactive complète

## 📚 Ressources

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

**Bon apprentissage MLOps ! 🚀**

Pour toute question, testez d'abord avec Swagger : http://localhost:8000/docs
