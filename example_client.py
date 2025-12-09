"""
Example client script for Drug Classification API
This demonstrates how to use the API to make predictions
"""

import requests
import json


def predict_drug(patient_data):
    """
    Make a prediction for a single patient
    
    Args:
        patient_data (dict): Patient information
        
    Returns:
        dict: Prediction response
    """
    url = "http://localhost:8000/predict"
    
    try:
        response = requests.post(url, json=patient_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def main():
    """Example usage of the Drug Classification API"""
    
    print("=" * 60)
    print("Drug Classification API - Example Client")
    print("=" * 60)
    
    # Example 1: High blood pressure patient
    print("\n--- Example 1: High BP Patient ---")
    patient1 = {
        "Age": 45,
        "Sex": "M",
        "BP": "HIGH",
        "Cholesterol": "NORMAL",
        "Na_to_K": 15.5
    }
    
    print(f"Patient Data: {json.dumps(patient1, indent=2)}")
    result1 = predict_drug(patient1)
    
    if result1:
        print(f"\nPredicted Drug: {result1['prediction']}")
        print(f"Status: {result1['status']}")
    
    # Example 2: Young female with normal BP
    print("\n\n--- Example 2: Young Female, Normal BP ---")
    patient2 = {
        "Age": 25,
        "Sex": "F",
        "BP": "NORMAL",
        "Cholesterol": "NORMAL",
        "Na_to_K": 10.5
    }
    
    print(f"Patient Data: {json.dumps(patient2, indent=2)}")
    result2 = predict_drug(patient2)
    
    if result2:
        print(f"\nPredicted Drug: {result2['prediction']}")
        print(f"Status: {result2['status']}")
    
    # Example 3: Elderly patient with high cholesterol
    print("\n\n--- Example 3: Elderly, High Cholesterol ---")
    patient3 = {
        "Age": 68,
        "Sex": "M",
        "BP": "LOW",
        "Cholesterol": "HIGH",
        "Na_to_K": 25.0
    }
    
    print(f"Patient Data: {json.dumps(patient3, indent=2)}")
    result3 = predict_drug(patient3)
    
    if result3:
        print(f"\nPredicted Drug: {result3['prediction']}")
        print(f"Status: {result3['status']}")
    
    # Example 4: Batch prediction
    print("\n\n--- Example 4: Batch Prediction (3 patients) ---")
    patients = [patient1, patient2, patient3]
    
    url = "http://localhost:8000/predict/batch"
    try:
        response = requests.post(url, json=patients)
        response.raise_for_status()
        results = response.json()
        
        print(f"\nBatch Results:")
        for i, result in enumerate(results, 1):
            print(f"  Patient {i}: {result['prediction']}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nMake sure the API server is running!")
    print("Start it with: make serve")
    print("\nPress Enter to continue...")
    input()
    
    main()
