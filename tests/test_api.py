"""
Script de test pour l'API Drug Classification
Atelier 4 : Test des endpoints REST
"""

import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"


def test_health():
    """Test du endpoint /health"""
    print("\n" + "="*60)
    print("TEST 1 : Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Réponse: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def test_predict():
    """Test du endpoint /predict"""
    print("\n" + "="*60)
    print("TEST 2 : Prédiction Simple")
    print("="*60)
    
    # Données patient exemple
    patient = {
        "Age": 45,
        "Sex": "M",
        "BP": "HIGH",
        "Cholesterol": "NORMAL",
        "Na_to_K": 15.5
    }
    
    print(f"\nDonnées patient:")
    print(json.dumps(patient, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient
        )
        print(f"\n✓ Status Code: {response.status_code}")
        result = response.json()
        print(f"✓ Médicament prédit: {result.get('prediction', 'N/A')}")
        print(f"\nRéponse complète:")
        print(json.dumps(result, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def test_retrain():
    """Test du endpoint /retrain (Excellence)"""
    print("\n" + "="*60)
    print("TEST 3 : Ré-entraînement (Excellence)")
    print("="*60)
    
    # Paramètres de ré-entraînement
    params = {
        "n_estimators": 50,
        "max_leaf_nodes": 20,
        "test_size": 0.3,
        "apply_smote": True
    }
    
    print(f"\nParamètres de ré-entraînement:")
    print(json.dumps(params, indent=2))
    
    try:
        print("\n⚠ Attention: Le ré-entraînement peut prendre quelques secondes...")
        response = requests.post(
            f"{BASE_URL}/retrain",
            json=params
        )
        print(f"\n✓ Status Code: {response.status_code}")
        result = response.json()
        print(f"✓ Statut: {result.get('status', 'N/A')}")
        print(f"✓ Accuracy: {result.get('accuracy', 'N/A')}")
        print(f"\nRéponse complète:")
        print(json.dumps(result, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def test_model_info():
    """Test du endpoint /model/info"""
    print("\n" + "="*60)
    print("TEST 4 : Informations Modèle")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Informations modèle:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def run_tests():
    """Exécuter tous les tests"""
    print("\n" + "="*60)
    print("TESTS API - DRUG CLASSIFICATION")
    print("="*60)
    
    results = {
        "Health Check": test_health(),
        "Prédiction": test_predict(),
        "Info Modèle": test_model_info(),
        "Ré-entraînement (Excellence)": test_retrain()
    }
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests réussis")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n⚠ Assurez-vous que l'API est démarrée:")
    print("   make serve\n")
    
    try:
        # Vérifier que l'API est accessible
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("✓ API accessible\n")
        run_tests()
    except requests.exceptions.ConnectionError:
        print("✗ ERREUR: Impossible de se connecter à l'API")
        print("\n  Démarrez d'abord le serveur:")
        print("    make serve")
        print("\n  Puis dans un autre terminal:")
        print("    make test-api\n")
    except Exception as e:
        print(f"✗ Erreur: {e}\n")
