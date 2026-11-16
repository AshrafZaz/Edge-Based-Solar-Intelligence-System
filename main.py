"""
Edge-Based Solar Intelligence System - Main Execution Orchestrator
File: main.py

CLI entry point to execute the complete machine learning workflow:
- Step 1: Raw dataset inspection and schema analysis
- Step 2: Chronological preprocessing and leak-free feature engineering
- Step 3: Model 1 training (Random Forest Regressor with validation tuning)
- Step 4: Model 2 training (XGBoost Regressor with validation tuning)
- Step 5: Untouched test set evaluation, model comparison, and graph generation
- Step 6: Model deserialization and offline prediction verification
- Step 7: Automated multi-layer fault auditing

Usage:
  python main.py --all           # Run entire workflow end-to-end
  python main.py --inspect       # Run dataset inspection only
  python main.py --preprocess    # Run preprocessing only
  python main.py --train         # Train both RF and XGBoost
  python main.py --evaluate      # Evaluate models and generate graphs
  python main.py --verify        # Verify saved model artifacts
  python main.py --audit         # Run automated fault checker
  python main.py --dashboard     # Display instructions to launch Streamlit dashboard
"""

import os
import sys
import argparse

# Add workspace root and src directory to Python search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.data_inspection import inspect_raw_dataset
from src.preprocessing import run_preprocessing_pipeline
from src.train_random_forest import train_and_tune_random_forest
from src.train_xgboost import train_and_tune_xgboost
from src.evaluate_models import evaluate_and_compare_models
from src.verify_models import verify_all_models
from src.fault_checker import run_fault_audit

def run_pipeline():
    print("\n" + "=" * 80)
    print("STARTING COMPLETE END-TO-END WORKFLOW: EDGE-BASED SOLAR INTELLIGENCE SYSTEM")
    print("=" * 80 + "\n")

    print("[STEP 1/7] Inspecting raw dataset and generating initial EDA graphs...")
    inspect_raw_dataset()

    print("\n[STEP 2/7] Running preprocessing, lag creation, and chronological partitioning...")
    run_preprocessing_pipeline()

    print("\n[STEP 3/7] Training Model 1: Random Forest Regressor with validation tuning...")
    train_and_tune_random_forest()

    print("\n[STEP 4/7] Training Model 2: XGBoost Regressor with validation tuning...")
    train_and_tune_xgboost()

    print("\n[STEP 5/7] Evaluating both models on untouched test set and rendering graphs...")
    evaluate_and_compare_models()

    print("\n[STEP 6/7] Verifying model serialization and forward prediction capability...")
    verify_all_models()

    print("\n[STEP 7/7] Running automated fault audit across dataset, ML, and code...")
    run_fault_audit()

    print("\n" + "=" * 80)
    print("ALL WORKFLOW STAGES EXECUTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 80)
    print("To launch the interactive dashboard, run:")
    print("  streamlit run dashboard.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-Based Solar Intelligence System Workflow Runner")
    parser.add_argument("--all", action="store_true", help="Execute complete workflow end-to-end")
    parser.add_argument("--inspect", action="store_true", help="Run dataset inspection")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--train", action="store_true", help="Train RF and XGBoost")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models and render graphs")
    parser.add_argument("--verify", action="store_true", help="Verify model loading and predictions")
    parser.add_argument("--audit", action="store_true", help="Run automated fault checker")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard instructions")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_raw_dataset()
    elif args.preprocess:
        run_preprocessing_pipeline()
    elif args.train:
        train_and_tune_random_forest()
        train_and_tune_xgboost()
    elif args.evaluate:
        evaluate_and_compare_models()
    elif args.verify:
        verify_all_models()
    elif args.audit:
        run_fault_audit()
    elif args.dashboard:
        print("\nLaunch the Streamlit web dashboard with:")
        print("  .\\.venv\\Scripts\\streamlit.exe run dashboard.py\n")
    else:
        # Default action: run all
        run_pipeline()
