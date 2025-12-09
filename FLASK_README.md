# 🌐 Interface Web Flask - Drug Classification

Interface web moderne et intuitive pour consommer l'API Drug Classification FastAPI.

## ✨ Fonctionnalités

- **Page d'accueil** : Vue d'ensemble du système
- **Prédiction** : Interface conviviale pour faire des prédictions
- **Ré-entraînement** : Optimiser le modèle avec de nouveaux hyperparamètres
- **Informations Modèle** : Détails techniques du modèle
- **À propos** : Documentation du projet
- **Statut API** : Indicateur en temps réel de l'état de l'API

## 🚀 Démarrage

### Prérequis

1. **API FastAPI démarrée** (port 8000)
2. **Flask installé**

### Installation

```bash
# Installer Flask
pip install flask

# Ou via requirements.txt
make install
```

### Lancement

#### Option 1 : Via Makefile (Recommandé)

**Terminal 1** - Démarrer l'API :
```bash
make serve
```

**Terminal 2** - Démarrer Flask :
```bash
make flask
```

#### Option 2 : Commande Python directe

```bash
python flask_app.py
```

### Accès

- **Interface Flask** : http://localhost:5000
- **API FastAPI** : http://localhost:8000
- **Swagger UI** : http://localhost:8000/docs

## 📱 Pages Disponibles

### 1. Accueil (/)
- Vue d'ensemble du système
- Accès rapide aux fonctionnalités
- Statistiques du modèle

### 2. Prédiction (/predict)
- Formulaire de saisie des données patient
  - Âge (0-100 ans)
  - Sexe (M/F)
  - Pression Artérielle (HIGH/NORMAL/LOW)
  - Cholestérol (HIGH/NORMAL)
  - Ratio Na/K (valeur décimale)
- Affichage du résultat de prédiction
- Exemples de cas pré-définis

### 3. Ré-entraînement (/retrain)
- Configuration des hyperparamètres :
  - n_estimators (10-500)
  - max_leaf_nodes (2-100)
  - test_size (0.1-0.5)
  - apply_smote (on/off)
- Affichage de l'accuracy après ré-entraînement
- Guide des hyperparamètres

### 4. Informations Modèle (/model-info)
- Type de modèle
- Nombre d'estimateurs
- Nombre de features
- Classes prédites
- État de l'API
- Architecture du pipeline

### 5. À propos (/about)
- Objectifs du projet
- Technologies utilisées
- Architecture système
- Instructions de démarrage

## 🎨 Design

- **Framework CSS** : Bootstrap 5
- **Icônes** : Font Awesome 6
- **Thème** : Gradient moderne (violet/bleu)
- **Responsive** : Adapté mobile/tablette/desktop
- **Animations** : Transitions fluides
- **Indicateur d'état** : Statut API en temps réel

## 🔧 Structure des Fichiers

```
omar-jalled-4ds8-ml_project/
├── flask_app.py              # Application Flask principale
├── templates/                # Templates HTML
│   ├── base.html            # Template de base
│   ├── index.html           # Page d'accueil
│   ├── predict.html         # Page de prédiction
│   ├── retrain.html         # Page de ré-entraînement
│   ├── model_info.html      # Infos modèle
│   └── about.html           # À propos
└── static/                   # Fichiers statiques (vide, CDN utilisé)
```

## 📡 Communication avec l'API

Flask communique avec FastAPI via HTTP :

```python
# Exemple de prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json=patient_data
)
result = response.json()
```

## ⚙️ Configuration

Modifier l'URL de l'API dans `flask_app.py` :

```python
API_URL = "http://localhost:8000"  # URL de l'API FastAPI
```

## 🐛 Résolution de Problèmes

### Erreur : "API n'est pas accessible"

**Solution** : Démarrez d'abord l'API FastAPI
```bash
make serve
```

### Erreur : "Port 5000 déjà utilisé"

**Solution** : Modifiez le port dans `flask_app.py`
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Templates non trouvés

**Solution** : Vérifiez que le dossier `templates/` existe
```bash
ls -la templates/
```

## 🌟 Fonctionnalités Avancées

### Vérification d'état en temps réel

L'interface vérifie automatiquement l'état de l'API toutes les 5 secondes :

```javascript
// Affiche un indicateur visuel
setInterval(checkApiStatus, 5000);
```

### Messages Flash

Feedback utilisateur pour toutes les actions :
- ✓ Succès (vert)
- ✗ Erreur (rouge)

### Validation côté client

Formulaires HTML5 avec validation :
- Champs requis
- Plages de valeurs (min/max)
- Types de données

## 📊 Exemples d'Utilisation

### Faire une prédiction

1. Accédez à http://localhost:5000/predict
2. Remplissez le formulaire
3. Cliquez sur "Prédire le Médicament"
4. Voyez le résultat affiché

### Ré-entraîner le modèle

1. Accédez à http://localhost:5000/retrain
2. Ajustez les hyperparamètres
3. Cliquez sur "Ré-entraîner le Modèle"
4. Attendez quelques secondes
5. L'accuracy s'affiche

## 🎯 Points Clés

- ✅ Interface moderne et intuitive
- ✅ Communication avec FastAPI
- ✅ Gestion des erreurs
- ✅ Responsive design
- ✅ Feedback en temps réel
- ✅ Documentation intégrée

## 📚 Technologies

- **Flask 3.x** : Framework web Python
- **Bootstrap 5** : Framework CSS
- **Font Awesome 6** : Bibliothèque d'icônes
- **Requests** : Client HTTP Python
- **Jinja2** : Moteur de templates

---

**Bon développement ! 🚀**

Pour toute question, consultez la page "À propos" dans l'interface.
