"""
Solar Intelligence System - Model Loading & Prediction Verification Script
File: src/verify_models.py

Performs offline verification that all serialized artifacts:
1. models/random_forest_model.pkl
2. models/xgboost_model.pkl
3. models/best_model.pkl
4. models/model_metadata.json
can be cleanly deserialized, validated, and used for forward inference.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path for standalone and package execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing import COMMON_FEATURES, TARGET_COL

def verify_all_models(models_dir: str = "models", processed_dir: str = "data/processed"):
    print("=" * 70)
    print("VERIFYING MODEL ARTIFACTS AND OFFLINE PREDICTIONS")
    print("=" * 70)
    
    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found at {test_path}")
    test_df = pd.read_csv(test_path)
    X_sample = test_df[COMMON_FEATURES].iloc[:5]
    y_actual = test_df[TARGET_COL].iloc[:5]
    
    print(f"[*] Loaded 5 verification test samples from '{test_path}'")
    
    # 1. Verify Random Forest
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    assert os.path.exists(rf_path), f"Missing {rf_path}"
    rf_artifact = joblib.load(rf_path)
    rf_model = rf_artifact['model']
    rf_preds = rf_model.predict(X_sample)
    print(f"[+] [PASS] Random Forest deserialized successfully. 5 Predictions: {np.round(rf_preds, 2)} W")
    
    # 2. Verify XGBoost
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    assert os.path.exists(xgb_path), f"Missing {xgb_path}"
    xgb_artifact = joblib.load(xgb_path)
    xgb_model = xgb_artifact['model']
    xgb_preds = xgb_model.predict(X_sample)
    print(f"[+] [PASS] XGBoost deserialized successfully.       5 Predictions: {np.round(xgb_preds, 2)} W")
    
    # 3. Verify Best Model
    best_path = os.path.join(models_dir, "best_model.pkl")
    assert os.path.exists(best_path), f"Missing {best_path}"
    best_artifact = joblib.load(best_path)
    best_model = best_artifact['model']
    best_preds = best_model.predict(X_sample)
    print(f"[+] [PASS] Best Model ('{best_artifact['model_name']}') deserialized successfully.")
    print(f"           5 Predictions: {np.round(best_preds, 2)} W")
    
    # 4. Verify Model Metadata
    meta_path = os.path.join(models_dir, "model_metadata.json")
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[+] [PASS] model_metadata.json valid. Selected Best: {meta['selected_best_model']}")
    print(f"           Input Features Count: {len(meta['input_features'])}")
    print(f"           Forecast Horizon: {meta['forecast_horizon']}")
    
    # Check predictions are plausible
    for i in range(len(X_sample)):
        print(f"  Sample {i+1}: Actual Future Power={y_actual.iloc[i]:.2f}W | RF Pred={rf_preds[i]:.2f}W | XGB Pred={xgb_preds[i]:.2f}W")
        assert 100.0 < rf_preds[i] < 400.0, f"Unrealistic RF prediction: {rf_preds[i]}"
        assert 100.0 < xgb_preds[i] < 400.0, f"Unrealistic XGB prediction: {xgb_preds[i]}"
        
    print("\n[+] ALL MODEL ARTIFACT AND PREDICTION VERIFICATION CHECKS PASSED!")
    return True

if __name__ == "__main__":
    verify_all_models()

