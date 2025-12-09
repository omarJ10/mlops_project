"""
FastAPI Application for Drug Classification Model
Atelier 4 : Exposition de la Fonction Predict via FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from model_pipeline import prepare_data, load_data, train_model, save_model

# Initialiser l'application FastAPI
app = FastAPI(
    title="Drug Classification API",
    description="API pour prédire les prescriptions de médicaments",
    version="1.0.0"
)

# Chemin du modèle
MODEL_PATH = "models/random_forest_model.pkl"

# Variables globales
model = None
feature_columns = None


class PatientData(BaseModel):
    """Schéma pour les données patient"""
    Age: int
    Sex: str
    BP: str
    Cholesterol: str
    Na_to_K: float

    class Config:
        schema_extra = {
            "example": {
                "Age": 45,
                "Sex": "M",
                "BP": "HIGH",
                "Cholesterol": "NORMAL",
                "Na_to_K": 15.5
            }
        }


class RetrainRequest(BaseModel):
    """Schéma pour le re-entrainement (Excellence)"""
    n_estimators: int = 100
    max_leaf_nodes: int = 30
    test_size: float = 0.3
    apply_smote: bool = False

    class Config:
        schema_extra = {
            "example": {
                "n_estimators": 150,
                "max_leaf_nodes": 40,
                "test_size": 0.3,
                "apply_smote": False
            }
        }


def preprocess_input(patient_data: PatientData) -> pd.DataFrame:
    """
    Prétraiter les données patient pour la prédiction
    """
    # Créer DataFrame
    df = pd.DataFrame([{
        'Age': patient_data.Age,
        'Sex': patient_data.Sex,
        'BP': patient_data.BP,
        'Cholesterol': patient_data.Cholesterol,
        'Na_to_K': patient_data.Na_to_K
    }])
    
    # Binning Age
    bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
    category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
    df['Age_binned'] = pd.cut(df['Age'], bins=bin_age, labels=category_age, include_lowest=True)
    df = df.drop(['Age'], axis=1)
    
    # Binning Na_to_K
    bin_NatoK = [0, 9, 19, 29, 50]
    category_NatoK = ['<10', '10-20', '20-30', '>30']
    df['Na_to_K_binned'] = pd.cut(df['Na_to_K'], bins=bin_NatoK, labels=category_NatoK, include_lowest=True)
    df = df.drop(['Na_to_K'], axis=1)
    
    # One-hot encoding
    df = pd.get_dummies(df)
    
    # Aligner avec les colonnes du modèle
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    
    return df


def load_model_and_features():
    """Charger le modèle et les features depuis le disque"""
    global model, feature_columns
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle non trouvé : {MODEL_PATH}. "
            "Entraînez d'abord le modèle avec: make pipeline"
        )
    
    try:
        # Charger le modèle
        model = joblib.load(MODEL_PATH)
        print(f"✓ Modèle chargé depuis {MODEL_PATH}")
        
        # Charger les colonnes de features
        if os.path.exists('data/prepared_data.pkl'):
            X_train, _, _, _ = joblib.load('data/prepared_data.pkl')
            feature_columns = X_train.columns.tolist()
            print(f"✓ Colonnes chargées : {len(feature_columns)} features")
        else:
            # Créer les features en préparant un échantillon
            print("⚠ Extraction des features depuis les données...")
            df = load_data('drug200.csv')
            if df is not None:
                X_train, _, _, _ = prepare_data(df, apply_smote=False)
                feature_columns = X_train.columns.tolist()
                print(f"✓ Features extraites : {len(feature_columns)} colonnes")
        
    except Exception as e:
        raise RuntimeError(f"Erreur chargement modèle: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    try:
        load_model_and_features()
        print("✓ API démarrée avec succès")
    except Exception as e:
        print(f"✗ Erreur démarrage: {str(e)}")


@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Drug Classification API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST) - Excellence"
        }
    }


@app.get("/health")
async def health_check():
    """Vérifier l'état de l'API et du modèle"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }


@app.post("/predict")
async def predict(patient_data: PatientData):
    """
    Route POST pour effectuer une prédiction
    
    Utilise le modèle pour prédire le médicament approprié
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Entraînez le modèle avec: make pipeline"
        )
    
    try:
        # Prétraiter les données
        processed_data = preprocess_input(patient_data)
        
        # Faire la prédiction
        prediction = model.predict(processed_data)
        predicted_drug = prediction[0]
        
        return {
            "prediction": predicted_drug,
            "patient_data": patient_data.dict(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/retrain")
async def retrain(params: RetrainRequest):
    """
    Route POST pour ré-entraîner le modèle (Excellence)
    
    Permet de ré-entraîner le modèle avec de nouveaux hyperparamètres
    """
    global model, feature_columns
    
    try:
        print(f"\n{'='*50}")
        print("RE-ENTRAINEMENT DU MODELE")
        print(f"{'='*50}")
        print(f"Paramètres:")
        print(f"  - n_estimators: {params.n_estimators}")
        print(f"  - max_leaf_nodes: {params.max_leaf_nodes}")
        print(f"  - test_size: {params.test_size}")
        print(f"  - apply_smote: {params.apply_smote}")
        
        # Charger les données
        df = load_data('drug200.csv')
        if df is None:
            raise HTTPException(
                status_code=500,
                detail="Impossible de charger les données"
            )
        
        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            test_size=params.test_size,
            apply_smote=params.apply_smote
        )
        
        # Sauvegarder les features
        feature_columns = X_train.columns.tolist()
        
        # Entraîner le modèle
        new_model = train_model(
            X_train, y_train,
            n_estimators=params.n_estimators,
            max_leaf_nodes=params.max_leaf_nodes
        )
        
        # Sauvegarder le modèle
        os.makedirs('models', exist_ok=True)
        save_model(new_model, MODEL_PATH)
        
        # Mettre à jour le modèle global
        model = new_model
        
        # Calculer l'accuracy
        accuracy = new_model.score(X_test, y_test)
        
        print(f"\n✓ Modèle ré-entraîné avec succès!")
        print(f"✓ Accuracy: {accuracy*100:.2f}%")
        
        return {
            "status": "success",
            "message": "Modèle ré-entraîné avec succès",
            "accuracy": f"{accuracy*100:.2f}%",
            "parameters": params.dict(),
            "model_path": MODEL_PATH
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du ré-entrainement: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Obtenir des informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    info = {
        "model_type": type(model).__name__,
        "n_features": len(feature_columns) if feature_columns else None,
        "model_path": MODEL_PATH
    }
    
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    
    if hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist()
    
    return info


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Démarrage du serveur FastAPI")
    print("="*60)
    print("API disponible sur: http://0.0.0.0:8000")
    print("Documentation interactive: http://0.0.0.0:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
