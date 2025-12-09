"""
Drug Classification Model Pipeline
This module contains modular functions for the drug classification ML pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("drug_classification_experiment")


def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing the drug dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Data loaded successfully from {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading data: {str(e)}")
        return None


def explore_data(df):
    """
    Perform initial data exploration and display basic statistics.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Dictionary containing exploration results.
    """
    print("\n" + "=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)

    exploration_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_values': df.isnull().sum().to_dict(),
        'drug_distribution': df['Drug'].value_counts().to_dict() if 'Drug' in df.columns else None
    }

    print(f"\n[OK] Dataset shape: {df.shape}")
    print(f"\n[OK] Column types:")
    print(df.dtypes)
    print(f"\n[OK] Missing values:")
    print(df.isnull().sum())
    print(f"\n[OK] Basic statistics:")
    print(df.describe())

    if 'Drug' in df.columns:
        print(f"\n[OK] Drug distribution:")
        print(df['Drug'].value_counts())

    return exploration_results


def prepare_data(df, test_size=0.3, random_state=0, apply_smote=True):
    """
    Preprocess the data: binning, encoding, splitting, and SMOTE.

    Args:
        df (pd.DataFrame): Raw dataset.
        test_size (float): Proportion of test set (default: 0.3).
        random_state (int): Random state for reproducibility (default: 0).
        apply_smote (bool): Whether to apply SMOTE oversampling (default: True).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 50)
    print("DATA PREPARATION")
    print("=" * 50)

    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # 1. Data Binning for Age
    print("\n[OK] Applying age binning...")
    bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
    category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
    df_processed['Age_binned'] = pd.cut(
        df_processed['Age'],
        bins=bin_age,
        labels=category_age,
        include_lowest=True)
    df_processed = df_processed.drop(['Age'], axis=1)

    # 2. Data Binning for Na_to_K
    print("[OK] Applying Na_to_K binning...")
    bin_NatoK = [0, 9, 19, 29, 50]
    category_NatoK = ['<10', '10-20', '20-30', '>30']
    df_processed['Na_to_K_binned'] = pd.cut(
        df_processed['Na_to_K'],
        bins=bin_NatoK,
        labels=category_NatoK,
        include_lowest=True)
    df_processed = df_processed.drop(['Na_to_K'], axis=1)

    # 3. Split features and target
    print("[OK] Splitting features and target...")
    X = df_processed.drop(["Drug"], axis=1)
    y = df_processed["Drug"]

    # 4. Train-test split
    print(f"[OK] Splitting data (train: {int((1 - test_size) * 100)}%, test: {int(test_size * 100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. One-hot encoding
    print("[OK] Applying one-hot encoding...")
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Ensure both train and test have the same columns
    # Get missing columns in the training test
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    # Ensure the order of column in the test set is in the same order than in
    # train set
    X_test = X_test[X_train.columns]

    # 6. Apply SMOTE
    if apply_smote:
        print("[OK] Applying SMOTE for class balancing...")
        X_train, y_train = SMOTE(
            random_state=random_state).fit_resample(
            X_train, y_train)
        print(f"  Training set shape after SMOTE: {X_train.shape}")
        print(f"  Class distribution after SMOTE:")
        print(y_train.value_counts())

    print(
        f"\n[OK] Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"                 y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(
        X_train,
        y_train,
        n_estimators=100,
        max_leaf_nodes=30,
        random_state=1,
        **kwargs):
    """
    Train a Random Forest classifier with MLflow tracking.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of trees in the forest (default: 100).
        max_leaf_nodes (int): Maximum number of leaf nodes (default: 30).
        random_state (int): Random state for reproducibility (default: 1).
        **kwargs: Additional parameters for RandomForestClassifier.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    print("\n" + "=" * 50)
    print("TRAINING MODEL: RANDOM FOREST")
    print("=" * 50)

    # Démarrer un run MLflow
    with mlflow.start_run(run_name="RandomForest_Training"):
        # Log des hyperparamètres
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("training_samples", len(X_train))
        
        # Log des kwargs additionnels
        for key, value in kwargs.items():
            mlflow.log_param(key, value)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            **kwargs
        )

        print(f"\n[OK] Training Random Forest model...")
        print(f"    - n_estimators: {n_estimators}")
        print(f"    - max_leaf_nodes: {max_leaf_nodes}")
        print(f"    - random_state: {random_state}")
        model.fit(X_train, y_train)
        print(f"[OK] Model trained successfully!")
        
        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        print(f"[OK] Model logged to MLflow")

    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate the trained model on test data with MLflow tracking.

    Args:
        model (object): Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for display purposes.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    print("\n" + "=" * 50)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("=" * 50)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Calculer des métriques supplémentaires
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log des métriques dans MLflow (cherche un run actif)
    if mlflow.active_run():
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_samples", len(X_test))
        print(f"[OK] Metrics logged to MLflow")

    # Display results
    print(f"\n[OK] Classification Report:")
    print(class_report)
    print(f"\n[OK] Confusion Matrix:")
    print(conf_matrix)
    print(f"\n[OK] Accuracy: {accuracy * 100:.2f}%")
    print(f"[OK] Precision: {precision * 100:.2f}%")
    print(f"[OK] Recall: {recall * 100:.2f}%")
    print(f"[OK] F1-Score: {f1 * 100:.2f}%")

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred
    }

    return results


def save_model(model, filepath='model.pkl'):
    """
    Save a trained model to disk using joblib.

    Args:
        model (object): Trained machine learning model.
        filepath (str): Path where the model will be saved (default: 'model.pkl').

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(
            filepath) else '.', exist_ok=True)

        joblib.dump(model, filepath)
        print(f"\n[OK] Model saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Error saving model: {str(e)}")
        return False


def load_model(filepath='model.pkl'):
    """
    Load a saved model from disk using joblib.

    Args:
        filepath (str): Path to the saved model file (default: 'model.pkl').

    Returns:
        object: Loaded model, or None if loading fails.
    """
    try:
        model = joblib.load(filepath)
        print(f"\n[OK] Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        print(f"\n[ERROR] Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"\n[ERROR] Error loading model: {str(e)}")
        return None


def compare_models(results_dict):
    """
    Compare multiple models based on their accuracy scores.

    Args:
        results_dict (dict): Dictionary with model names as keys and their results as values.
                            Each result should contain an 'accuracy' key.

    Returns:
        pd.DataFrame: Comparison dataframe sorted by accuracy.
    """
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'] * 100
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(
        by='Accuracy',
        ascending=False).reset_index(
        drop=True)

    print("\n[OK] Model Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def predict_new_data(model, X_new):
    """
    Make predictions on new data using a trained model.

    Args:
        model (object): Trained machine learning model.
        X_new (pd.DataFrame): New data for prediction (must be preprocessed).

    Returns:
        np.ndarray: Predictions for the new data.
    """
    try:
        predictions = model.predict(X_new)
        print(
            f"\n[OK] Predictions made successfully for {len(predictions)} samples")
        return predictions
    except Exception as e:
        print(f"\n[ERROR] Error making predictions: {str(e)}")
        return None


def visualize_results(y_test, y_pred, model_name='Model'):
    """
    Visualize model results with confusion matrix heatmap.

    Args:
        y_test (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Name of the model for the plot title.
    """
    plt.figure(figsize=(10, 7))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the plot
    plot_filename = f'{model_name.replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(plot_filename)
    print(f"\n[OK] Confusion matrix visualization saved to {plot_filename}")
    plt.close()


def optimize_hyperparameters(
        X_train,
        X_test,
        y_train,
        y_test,
        param_name='max_leaf_nodes',
        param_range=None):
    """
    Optimize hyperparameters for Random Forest with MLflow tracking.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        param_name (str): Parameter to optimize ('max_leaf_nodes' or 'n_estimators').
        param_range (range or list): Range of parameters to test.

    Returns:
        dict: Dictionary with best parameters and score.
    """
    print("\n" + "=" * 50)
    print(f"HYPERPARAMETER OPTIMIZATION: RANDOM FOREST ({param_name})")
    print("=" * 50)

    if param_range is None:
        param_range = range(
            2, 50) if param_name == 'max_leaf_nodes' else range(
            50, 300, 25)

    scores = []

    # Run parent pour l'optimisation
    with mlflow.start_run(run_name=f"Hyperparameter_Optimization_{param_name}"):
        mlflow.log_param("optimization_param", param_name)
        mlflow.log_param("param_range_start", list(param_range)[0])
        mlflow.log_param("param_range_end", list(param_range)[-1])

        for param_value in param_range:
            # Run enfant pour chaque valeur testée
            with mlflow.start_run(run_name=f"{param_name}_{param_value}", nested=True):
                if param_name == 'max_leaf_nodes':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=1,
                        max_leaf_nodes=param_value)
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_param("max_leaf_nodes", param_value)
                elif param_name == 'n_estimators':
                    model = RandomForestClassifier(
                        n_estimators=param_value,
                        random_state=1,
                        max_leaf_nodes=30)
                    mlflow.log_param("n_estimators", param_value)
                    mlflow.log_param("max_leaf_nodes", 30)
                else:
                    raise ValueError(
                        f"Invalid param_name. Choose 'max_leaf_nodes' or 'n_estimators'")

                mlflow.log_param("random_state", 1)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
                
                # Log la métrique pour cette valeur
                mlflow.log_metric("test_accuracy", score)
                mlflow.log_metric(param_name, param_value)

        best_idx = np.argmax(scores)
        best_param = list(param_range)[best_idx]
        best_score = scores[best_idx]

        # Log les meilleurs résultats dans le run parent
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.log_param(f"best_{param_name}", best_param)

        print(f"\n[OK] Best {param_name}: {best_param}")
        print(f"[OK] Best accuracy: {best_score * 100:.2f}%")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(list(param_range), scores, marker='o', linewidth=2, markersize=6)
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel('Accuracy Score')
        plt.title(
            f'Random Forest - {param_name.replace("_", " ").title()} Optimization')
        plt.grid(True, alpha=0.3)
        plot_filename = f'random_forest_{param_name}_optimization.png'
        plt.savefig(plot_filename)
        print(f"[OK] Optimization plot saved to {plot_filename}")
        
        # Log le plot dans MLflow
        mlflow.log_artifact(plot_filename)
        plt.close()

    return {
        'best_param': best_param,
        'best_score': best_score,
        'param_name': param_name,
        'all_scores': scores
    }
