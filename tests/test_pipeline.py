"""
Test script for the drug classification ML pipeline.
This script validates that all functions work correctly.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Import pipeline functions
from model_pipeline import (
    load_data, explore_data, prepare_data, train_model,
    evaluate_model, save_model, load_model
)


def test_load_data():
    """Test data loading functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Load Data")
    print("=" * 70)

    try:
        df = load_data('drug200.csv')
        assert df is not None, "Data loading failed"
        assert df.shape[0] > 0, "Dataset is empty"
        assert 'Drug' in df.columns, "Target column 'Drug' not found"
        print("[OK] ✓ Data loading test passed!")
        return True
    except Exception as e:
        print(f"[ERROR] ✗ Data loading test failed: {str(e)}")
        return False


def test_explore_data():
    """Test data exploration functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: Explore Data")
    print("=" * 70)

    try:
        df = load_data('drug200.csv')
        results = explore_data(df)
        assert results is not None, "Exploration failed"
        assert 'shape' in results, "Shape information missing"
        assert 'columns' in results, "Column information missing"
        print("[OK] ✓ Data exploration test passed!")
        return True
    except Exception as e:
        print(f"[ERROR] ✗ Data exploration test failed: {str(e)}")
        return False


def test_prepare_data():
    """Test data preparation functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: Prepare Data")
    print("=" * 70)

    try:
        df = load_data('drug200.csv')
        X_train, X_test, y_train, y_test = prepare_data(df, apply_smote=False)

        assert X_train.shape[0] > 0, "Training set is empty"
        assert X_test.shape[0] > 0, "Test set is empty"
        assert len(y_train) == X_train.shape[0], "Mismatch in training data"
        assert len(y_test) == X_test.shape[0], "Mismatch in test data"

        print("[OK] ✓ Data preparation test passed!")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"[ERROR] ✗ Data preparation test failed: {str(e)}")
        return None, None, None, None


def test_train_model(X_train, y_train):
    """Test model training functionality."""
    print("\n" + "=" * 70)
    print("TEST 4: Train Model")
    print("=" * 70)

    try:
        model = train_model(
            X_train,
            y_train,
            n_estimators=50,
            max_leaf_nodes=20)
        assert model is not None, "Model training failed"
        assert hasattr(model, 'predict'), "Model doesn't have predict method"

        print("[OK] ✓ Model training test passed!")
        return model
    except Exception as e:
        print(f"[ERROR] ✗ Model training test failed: {str(e)}")
        return None


def test_evaluate_model(model, X_test, y_test):
    """Test model evaluation functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Evaluate Model")
    print("=" * 70)

    try:
        results = evaluate_model(model, X_test, y_test, 'Random Forest')
        assert results is not None, "Evaluation failed"
        assert 'accuracy' in results, "Accuracy not in results"
        assert results['accuracy'] > 0, "Invalid accuracy score"

        print("[OK] ✓ Model evaluation test passed!")
        return results
    except Exception as e:
        print(f"[ERROR] ✗ Model evaluation test failed: {str(e)}")
        return None


def test_save_load_model(model):
    """Test model saving and loading functionality."""
    print("\n" + "=" * 70)
    print("TEST 6: Save and Load Model")
    print("=" * 70)

    try:
        # Create test directory
        test_model_path = 'test_models/test_model.pkl'
        os.makedirs('test_models', exist_ok=True)

        # Test saving
        save_result = save_model(model, test_model_path)
        assert save_result, "Model saving failed"
        assert os.path.exists(test_model_path), "Model file not created"

        # Test loading
        loaded_model = load_model(test_model_path)
        assert loaded_model is not None, "Model loading failed"
        assert hasattr(loaded_model, 'predict'), "Loaded model invalid"

        # Clean up
        os.remove(test_model_path)
        os.rmdir('test_models')

        print("[OK] ✓ Model save/load test passed!")
        return True
    except Exception as e:
        print(f"[ERROR] ✗ Model save/load test failed: {str(e)}")
        return False


def test_prediction(model, X_test):
    """Test prediction functionality."""
    print("\n" + "=" * 70)
    print("TEST 7: Make Predictions")
    print("=" * 70)

    try:
        predictions = model.predict(X_test)
        assert predictions is not None, "Prediction failed"
        assert len(predictions) == X_test.shape[0], "Prediction count mismatch"

        print(
            f"[OK] ✓ Prediction test passed! Made {
                len(predictions)} predictions")
        return True
    except Exception as e:
        print(f"[ERROR] ✗ Prediction test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RUNNING ALL TESTS")
    print("=" * 70)

    test_results = {}

    # Test 1: Load Data
    test_results['load_data'] = test_load_data()

    # Test 2: Explore Data
    test_results['explore_data'] = test_explore_data()

    # Test 3: Prepare Data
    X_train, X_test, y_train, y_test = test_prepare_data()
    test_results['prepare_data'] = (X_train is not None)

    if X_train is not None:
        # Test 4: Train Model
        model = test_train_model(X_train, y_train)
        test_results['train_model'] = (model is not None)

        if model is not None:
            # Test 5: Evaluate Model
            results = test_evaluate_model(model, X_test, y_test)
            test_results['evaluate_model'] = (results is not None)

            # Test 6: Save/Load Model
            test_results['save_load_model'] = test_save_load_model(model)

            # Test 7: Predictions
            test_results['prediction'] = test_prediction(model, X_test)

    # Display final results
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests

    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} : {status}")

    print("=" * 70)
    print(
        f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")

    if failed_tests == 0:
        print("\n[OK] ✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 70)
        return 0
    else:
        print(f"\n[ERROR] ✗✗✗ {failed_tests} TEST(S) FAILED ✗✗✗")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
