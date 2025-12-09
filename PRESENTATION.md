# 🎉 Drug Classification System - Présentation Complète

## 📋 Vue d'Ensemble

Système complet de **prédiction de médicaments** avec Machine Learning, exposé via **API REST (FastAPI)** et **Interface Web (Flask)**.

---

## 🏗️ Architecture du Système

```
┌─────────────────┐
│   Utilisateur   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Flask Web Interface       │
│   (Port 5000)               │
│   - Interface graphique     │
│   - Formulaires intuitifs   │
└────────┬────────────────────┘
         │ HTTP Requests
         ▼
┌─────────────────────────────┐
│   FastAPI REST API          │
│   (Port 8000)               │
│   - /predict                │
│   - /retrain (Excellence)   │
│   - /model/info             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Random Forest Model       │
│   (100 arbres)              │
│   - 18 features             │
│   - 5 classes prédites      │
└─────────────────────────────┘
```

---

## 📦 Fichiers du Projet

### 🔹 Backend & ML
| Fichier | Description |
|---------|-------------|
| `app.py` | Application FastAPI (API REST) |
| `flask_app.py` | Interface web Flask |
| `model_pipeline.py` | Pipeline ML (entraînement, prétraitement) |
| `main.py` | Script CLI pour le pipeline |

### 🔹 Templates & Interface
| Dossier/Fichier | Description |
|-----------------|-------------|
| `templates/base.html` | Template de base (Bootstrap 5) |
| `templates/index.html` | Page d'accueil |
| `templates/predict.html` | Page de prédiction |
| `templates/retrain.html` | Page de ré-entraînement |
| `templates/model_info.html` | Informations modèle |
| `templates/about.html` | À propos du projet |

### 🔹 Tests & Configuration
| Fichier | Description |
|---------|-------------|
| `test_api.py` | Tests automatisés API |
| `test_pipeline.py` | Tests pipeline ML |
| `Makefile` | Commandes automatisées |
| `requirements.txt` | Dépendances Python |

### 🔹 Documentation
| Fichier | Description |
|---------|-------------|
| `ATELIER4_README.md` | Documentation Atelier 4 |
| `FLASK_README.md` | Guide interface Flask |
| `PRESENTATION.md` | Ce fichier |

### 🔹 Données & Modèles
| Fichier/Dossier | Description |
|-----------------|-------------|
| `drug200.csv` | Dataset (200 patients) |
| `models/random_forest_model.pkl` | Modèle entraîné (643 KB) |

---

## 🚀 Démarrage Rapide

### Étape 1 : Installation

```bash
# Cloner le projet (si besoin)
cd ~/omar-jalled-4ds8-ml_project

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances
make install
```

### Étape 2 : Entraîner le Modèle

```bash
# Entraîner le modèle Random Forest
make pipeline
```

### Étape 3 : Démarrer l'API

**Terminal 1** :
```bash
make serve
```

Sortie :
```
═══════════════════════════════════════════════════════════
  Démarrage du serveur FastAPI
═══════════════════════════════════════════════════════════

  API disponible sur : http://localhost:8000
  Documentation Swagger : http://localhost:8000/docs
```

### Étape 4 : Démarrer l'Interface Web

**Terminal 2** :
```bash
make flask
```

Sortie :
```
═══════════════════════════════════════════════════════════
  Démarrage de l'interface Flask
═══════════════════════════════════════════════════════════

  Interface web : http://localhost:5000
```

### Étape 5 : Utiliser le Système

Ouvrez votre navigateur : **http://localhost:5000**

---

## 🌟 Fonctionnalités Principales

### 1. 🧠 Prédiction de Médicaments

**Entrée (Données Patient)** :
```json
{
  "Age": 45,
  "Sex": "M",
  "BP": "HIGH",
  "Cholesterol": "NORMAL",
  "Na_to_K": 15.5
}
```

**Sortie (Prédiction)** :
```json
{
  "prediction": "drugX",
  "status": "success"
}
```

**Accès** :
- Interface : http://localhost:5000/predict
- API : http://localhost:8000/predict

### 2. 🔄 Ré-entraînement du Modèle (Excellence)

Permet d'optimiser le modèle avec de nouveaux hyperparamètres :

**Paramètres** :
- `n_estimators` : 10-500 (nombre d'arbres)
- `max_leaf_nodes` : 2-100 (profondeur)
- `test_size` : 0.1-0.5 (proportion test)
- `apply_smote` : true/false (rééquilibrage)

**Accès** :
- Interface : http://localhost:5000/retrain
- API : http://localhost:8000/retrain

### 3. ℹ️ Informations Modèle

Affiche les détails techniques :
- Type : RandomForestClassifier
- N_estimators : 100
- Features : 18
- Classes : drugA, drugB, drugC, drugX, drugY

**Accès** :
- Interface : http://localhost:5000/model-info
- API : http://localhost:8000/model/info

### 4. 📚 Documentation Interactive (Swagger)

Documentation API auto-générée avec interface de test.

**Accès** : http://localhost:8000/docs

---

## 🎯 Points Forts du Projet

### ✅ Atelier 4 - Objectifs Atteints

- [x] Route `/predict` exposée via FastAPI
- [x] Chargement du modèle depuis le disque
- [x] Documentation Swagger interactive
- [x] Tests automatisés
- [x] Commande Makefile pour démarrer l'API
- [x] **Excellence** : Route `/retrain` pour ré-entraînement

### ✅ Bonus : Interface Web

- [x] Interface Flask moderne et intuitive
- [x] Design responsive (Bootstrap 5)
- [x] Formulaires de prédiction
- [x] Page de ré-entraînement
- [x] Monitoring en temps réel de l'API
- [x] Documentation complète

---

## 🛠️ Technologies Utilisées

### Backend
- **FastAPI** : Framework API moderne
- **Flask** : Framework web Python
- **Uvicorn** : Serveur ASGI performant
- **Pydantic** : Validation de données

### Machine Learning
- **Scikit-learn** : Random Forest
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **imbalanced-learn** : SMOTE

### Frontend
- **Bootstrap 5** : Framework CSS
- **Font Awesome 6** : Icônes
- **JavaScript** : Interactions dynamiques
- **Jinja2** : Moteur de templates

---

## 📊 Performance du Modèle

### Caractéristiques
- **Algorithme** : Random Forest
- **N_estimators** : 100 arbres
- **Max_leaf_nodes** : 30
- **Accuracy** : ~98%
- **Temps de prédiction** : < 50ms

### Dataset
- **Fichier** : drug200.csv
- **Taille** : 200 patients
- **Features** : 5 (Age, Sex, BP, Cholesterol, Na_to_K)
- **Classes** : 5 médicaments

---

## 🔧 Commandes Makefile

| Commande | Description |
|----------|-------------|
| `make install` | Installer les dépendances |
| `make pipeline` | Entraîner le modèle |
| `make serve` | Démarrer l'API FastAPI |
| `make flask` | Démarrer l'interface Flask |
| `make test-api` | Tester l'API |
| `make lint` | Vérifier le code |
| `make security` | Scanner la sécurité |
| `make clean` | Nettoyer les fichiers temporaires |

---

## 📸 Captures d'Écran

### Page d'Accueil
- Vue d'ensemble du système
- 4 cartes de fonctionnalités
- Design moderne avec gradient

### Page Prédiction
- Formulaire intuitif
- Validation en temps réel
- Affichage du résultat avec style

### Page Ré-entraînement
- Configuration des hyperparamètres
- Affichage de l'accuracy
- Guide des paramètres

### Page Informations
- Statistiques du modèle
- État de l'API
- Architecture du pipeline

---

## 🎓 Contexte Académique

**Cours** : MLOps - Machine Learning Operations  
**Atelier** : Atelier 4 - Exposition de la Fonction Predict via FastAPI  
**Année** : 2025  
**Excellence** : ✓ Implémenté (route `/retrain`)

---

## 🐛 Résolution de Problèmes

### Problème : Modèle non trouvé
```bash
make pipeline  # Entraîner le modèle
```

### Problème : Port déjà utilisé
```bash
# Tuer le processus sur le port 8000
lsof -ti:8000 | xargs kill -9
```

### Problème : API non accessible depuis Flask
```bash
# Vérifier que l'API tourne
curl http://localhost:8000/health
```

### Problème : Erreur NumPy avec SMOTE
```
Solution : Désactiver SMOTE dans le ré-entraînement
(déjà configuré par défaut dans l'interface)
```

---

## 📈 Évolutions Possibles

### Court Terme
- [ ] Ajouter plus de visualisations (graphiques)
- [ ] Historique des prédictions
- [ ] Export des résultats (PDF/CSV)

### Moyen Terme
- [ ] Base de données pour stocker les prédictions
- [ ] Authentification utilisateur
- [ ] API rate limiting

### Long Terme
- [ ] Déploiement cloud (AWS/Azure/GCP)
- [ ] CI/CD avec GitHub Actions
- [ ] Monitoring avec Prometheus/Grafana
- [ ] Support multi-modèles

---

## 🎉 Résumé

Ce projet démontre une **implémentation complète MLOps** :

1. ✅ **ML Pipeline** : Entraînement, évaluation, sauvegarde
2. ✅ **API REST** : FastAPI moderne et performante
3. ✅ **Interface Web** : Flask intuitive et responsive
4. ✅ **Documentation** : Swagger UI + README complets
5. ✅ **Tests** : Suite de tests automatisés
6. ✅ **Automatisation** : Makefile pour toutes les tâches
7. ✅ **Excellence** : Ré-entraînement en ligne

---

## 📞 Support

Pour toute question :
1. Consultez les README spécifiques
2. Testez avec Swagger UI (http://localhost:8000/docs)
3. Vérifiez les logs dans les terminaux

---

**Projet développé avec ❤️ pour l'apprentissage MLOps**

**Date** : Novembre 2025  
**Version** : 1.0.0  
**Statut** : ✅ Production Ready
