"""
Application Flask pour consommer l'API Drug Classification
Interface web pour faire des prédictions et gérer le modèle
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import requests
import json

app = Flask(__name__)
app.secret_key = 'drug-classification-secret-key-2025'

# Configuration de l'API
API_URL = "http://localhost:8000"


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Page de prédiction"""
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            patient_data = {
                'Age': int(request.form['age']),
                'Sex': request.form['sex'],
                'BP': request.form['bp'],
                'Cholesterol': request.form['cholesterol'],
                'Na_to_K': float(request.form['na_to_k'])
            }
            
            # Appeler l'API
            response = requests.post(f"{API_URL}/predict", json=patient_data)
            
            if response.status_code == 200:
                result = response.json()
                return render_template('predict.html', 
                                     result=result, 
                                     patient_data=patient_data)
            else:
                flash(f"Erreur API: {response.json().get('detail', 'Erreur inconnue')}", 'error')
                
        except requests.exceptions.ConnectionError:
            flash("Erreur: L'API n'est pas accessible. Démarrez-la avec 'make serve'", 'error')
        except Exception as e:
            flash(f"Erreur: {str(e)}", 'error')
    
    return render_template('predict.html')


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    """Page de ré-entraînement"""
    if request.method == 'POST':
        try:
            # Récupérer les hyperparamètres
            params = {
                'n_estimators': int(request.form['n_estimators']),
                'max_leaf_nodes': int(request.form['max_leaf_nodes']),
                'test_size': float(request.form['test_size']),
                'apply_smote': request.form.get('apply_smote') == 'on'
            }
            
            # Appeler l'API
            response = requests.post(f"{API_URL}/retrain", json=params)
            
            if response.status_code == 200:
                result = response.json()
                flash(f"✓ Modèle ré-entraîné avec succès! Accuracy: {result['accuracy']}", 'success')
                return render_template('retrain.html', result=result)
            else:
                flash(f"Erreur API: {response.json().get('detail', 'Erreur inconnue')}", 'error')
                
        except requests.exceptions.ConnectionError:
            flash("Erreur: L'API n'est pas accessible. Démarrez-la avec 'make serve'", 'error')
        except Exception as e:
            flash(f"Erreur: {str(e)}", 'error')
    
    return render_template('retrain.html')


@app.route('/model-info')
def model_info():
    """Page d'informations sur le modèle"""
    try:
        # Récupérer les infos du modèle
        response = requests.get(f"{API_URL}/model/info")
        health_response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            model_data = response.json()
            health_data = health_response.json()
            return render_template('model_info.html', 
                                 model=model_data, 
                                 health=health_data)
        else:
            flash("Erreur lors de la récupération des informations", 'error')
            
    except requests.exceptions.ConnectionError:
        flash("Erreur: L'API n'est pas accessible. Démarrez-la avec 'make serve'", 'error')
    except Exception as e:
        flash(f"Erreur: {str(e)}", 'error')
    
    return render_template('model_info.html')


@app.route('/about')
def about():
    """Page à propos du projet"""
    return render_template('about.html')


@app.route('/api/health')
def api_health():
    """Vérifier l'état de l'API (pour AJAX)"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return jsonify(response.json())
    except:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Application Web Flask - Drug Classification")
    print("="*60)
    print("\n  Interface web disponible sur: http://localhost:5000")
    print("  Assurez-vous que l'API FastAPI tourne sur le port 8000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
