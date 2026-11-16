"""
Solar Intelligence System - Random Forest Training Script
File: src/train_random_forest.py

Trains a Random Forest Regressor for 1-step ahead solar power forecasting:
- Uses chronological train partition for model fitting.
- Uses validation partition for hyperparameter tuning.
- Analyzes feature importances.
- Evaluates MAE, RMSE, R² on train and validation splits.
- Saves model to models/random_forest_model.pkl.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in sys.path for standalone and package execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing import COMMON_FEATURES, TARGET_COL

def train_and_tune_random_forest(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    random_state: int = 42
):
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 60)
    print("TRAINING MODEL 1: RANDOM FOREST REGRESSOR")
    print("=" * 60)
    
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df[COMMON_FEATURES]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[COMMON_FEATURES]
    y_val = val_df[TARGET_COL]
    
    print(f"[*] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"[*] Input Features ({len(COMMON_FEATURES)}): {COMMON_FEATURES}")
    
    # Grid of candidate hyperparameters for validation tuning
    param_grid = [
        {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 2},
    ]
    
    best_model = None
    best_val_rmse = float('inf')
    best_params = None
    
    print("\n[*] Performing Validation-Based Hyperparameter Tuning...")
    for params in param_grid:
        rf = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"  Params: {params} -> Val RMSE: {val_rmse:.3f} W, Val R²: {val_r2:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = rf
            best_params = params

    print(f"\n[+] Optimal Hyperparameters selected on Validation: {best_params}")
    
    # Evaluate best model on Train and Validation
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print("\n--- Model 1 Performance (Random Forest) ---")
    print(f"Train: MAE={train_mae:.3f} W | RMSE={train_rmse:.3f} W | R²={train_r2:.4f}")
    print(f"Val:   MAE={val_mae:.3f} W | RMSE={val_rmse:.3f} W | R²={val_r2:.4f}")
    
    # Feature Importances
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': COMMON_FEATURES,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nFeature Importances (Top 7):")
    for idx, row in fi_df.head(7).iterrows():
        print(f"  {row['Feature']:<16}: {row['Importance']*100:.2f}%")
        
    # Save Model
    model_save_path = os.path.join(models_dir, "random_forest_model.pkl")
    model_artifact = {
        'model': best_model,
        'features': COMMON_FEATURES,
        'target': TARGET_COL,
        'best_params': best_params,
        'metrics_train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
        'metrics_val': {'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2},
        'feature_importances': fi_df.to_dict(orient='records')
    }
    joblib.dump(model_artifact, model_save_path)
    print(f"[+] Random Forest model saved to: {model_save_path}")
    return model_artifact

if __name__ == "__main__":
    train_and_tune_random_forest()
