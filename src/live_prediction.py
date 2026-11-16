"""
Solar Intelligence System - Live Real-Time Forecasting Engine
File: src/live_prediction.py

Integrates live ESP32 hardware telemetry with trained ML models:
1. Enforces strict feature alignment (14 features, exact order, exact physical units).
2. Prevents fake lag initialization: requires a genuine 2-step warm-up period.
3. Rigorous validation of sensor telemetry (finite, non-negative, valid ranges).
4. Out-of-Distribution (OOD) Domain Compatibility Checking:
   - Detects when edge hardware (e.g. 6V panel, 120mA) operates outside the 
     training dataset distribution (~30-37V, ~6-8A).
   - Issues explicit domain mismatch warnings rather than silently extrapolating.
5. Predicts 1-step ahead (~1-hour horizon) solar power using loaded models.
"""

import os
import sys
import math
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd

# Setup paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing import COMMON_FEATURES, TARGET_COL
from src.serial_reader import ESP32SerialReader, list_available_ports, get_physical_usb_ports

# Training distribution boundaries computed from data/processed/train.csv
TRAIN_DOMAIN_BOUNDS = {
    'Voltage': (30.05, 36.96),       # Volts (V)
    'Current': (6.30, 8.39),          # Amperes (A)
    'Power': (189.01, 300.38),        # Watts (W)
    'Temperature': (24.0, 36.0),      # °C
    'Humidity': (24.0, 94.0)          # %
}

class LivePredictionEngine:
    """Real-time inference orchestrator with historical lag buffer and OOD detection."""
    
    def __init__(self, models_dir: str = "models", history_len: int = 10):
        self.models_dir = os.path.join(_PROJECT_ROOT, models_dir) if not os.path.isabs(models_dir) else models_dir
        self.history_len = history_len
        self.history = deque(maxlen=history_len)
        
        # Models
        self.best_artifact = None
        self.rf_artifact = None
        self.xgb_artifact = None
        self.load_models()
        
    def load_models(self):
        best_path = os.path.join(self.models_dir, "best_model.pkl")
        rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
        xgb_path = os.path.join(self.models_dir, "xgboost_model.pkl")
        
        if os.path.exists(best_path):
            self.best_artifact = joblib.load(best_path)
        if os.path.exists(rf_path):
            self.rf_artifact = joblib.load(rf_path)
        if os.path.exists(xgb_path):
            self.xgb_artifact = joblib.load(xgb_path)

    def extract_time_features(self, dt: Optional[datetime] = None) -> Dict[str, float]:
        """Extract cyclical solar time proxies from current datetime."""
        if dt is None:
            dt = datetime.now()
        dec_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        sin_time = float(np.sin(2.0 * np.pi * dec_hour / 24.0))
        cos_time = float(np.cos(2.0 * np.pi * dec_hour / 24.0))
        return {
            'decimal_hour': dec_hour,
            'sin_time': sin_time,
            'cos_time': cos_time
        }

    def validate_telemetry(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify numerical validity and physical boundaries of sensor readings."""
        required = ['temperature', 'humidity', 'voltage', 'current', 'power']
        for k in required:
            if k not in sample or sample[k] is None:
                return False, f"Missing telemetry field: '{k}'"
            val = sample[k]
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return False, f"Non-finite or invalid numeric value for '{k}': {val}"

        if sample['voltage'] < 0.0 or sample['current'] < 0.0 or sample['power'] < 0.0:
            return False, f"Negative electrical reading: V={sample['voltage']}V, I={sample['current']}A, P={sample['power']}W"

        return True, "Valid"

    def check_out_of_distribution(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Compares live telemetry with historical training dataset boundaries.
        Returns: (is_ood: bool, warnings: List[str])
        """
        warnings = []
        is_ood = False

        v = sample['voltage']
        i = sample['current']
        p = sample['power']

        v_min, v_max = TRAIN_DOMAIN_BOUNDS['Voltage']
        i_min, i_max = TRAIN_DOMAIN_BOUNDS['Current']
        p_min, p_max = TRAIN_DOMAIN_BOUNDS['Power']

        if v < v_min or v > v_max:
            is_ood = True
            warnings.append(f"Voltage ({v:.2f}V) is outside training domain [{v_min:.2f}V - {v_max:.2f}V].")
            
        if i < i_min or i > i_max:
            is_ood = True
            warnings.append(f"Current ({i:.3f}A) is outside training domain [{i_min:.2f}A - {i_max:.2f}A].")

        if p < p_min or p > p_max:
            is_ood = True
            warnings.append(f"Power ({p:.2f}W) is outside training domain [{p_min:.2f}W - {p_max:.2f}W].")

        return is_ood, warnings

    def process_sample(self, current_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a genuine telemetry sample from hardware or buffer.
        
        Returns:
            Dict containing:
            - 'status': 'READY', 'WARMING_UP', 'INVALID'
            - 'warmup_count': Number of genuine samples in buffer
            - 'features': DataFrame or None
            - 'predictions': Dict of predictions or None
            - 'is_ood': bool
            - 'ood_warnings': List[str]
            - 'message': Diagnostic message
        """
        # 1. Validate numerical inputs
        is_valid, val_msg = self.validate_telemetry(current_sample)
        if not is_valid:
            return {
                'status': 'INVALID',
                'message': val_msg,
                'warmup_count': len(self.history),
                'features': None,
                'predictions': None,
                'is_ood': False,
                'ood_warnings': []
            }

        # 2. Check warm-up: Do we have a genuine prior observation (t-1)?
        # During offline training, sample 0 has no lag and is dropped.
        # Live inference MUST NOT fabricate lag by using sample 0 as its own lag!
        if len(self.history) == 0:
            self.history.append(current_sample)
            return {
                'status': 'WARMING_UP',
                'message': 'Warm-up observation 1/2 recorded. Awaiting 2nd observation for genuine prior lag t-1.',
                'warmup_count': 1,
                'features': None,
                'predictions': None,
                'is_ood': False,
                'ood_warnings': []
            }

        # 3. Construct genuine lag features using real prior observation
        prev_sample = self.history[-1]
        time_feats = self.extract_time_features()

        power_lag_1 = float(prev_sample['power'])
        voltage_lag_1 = float(prev_sample['voltage'])
        current_lag_1 = float(prev_sample['current'])
        temp_lag_1 = float(prev_sample['temperature'])
        humidity_lag_1 = float(prev_sample['humidity'])
        power_trend = float(current_sample['power']) - power_lag_1

        feat_dict = {
            'Temperature': float(current_sample['temperature']),
            'Humidity': float(current_sample['humidity']),
            'Voltage': float(current_sample['voltage']),
            'Current': float(current_sample['current']),
            'Power': float(current_sample['power']),
            'power_lag_1': power_lag_1,
            'voltage_lag_1': voltage_lag_1,
            'current_lag_1': current_lag_1,
            'temp_lag_1': temp_lag_1,
            'humidity_lag_1': humidity_lag_1,
            'power_trend': power_trend,
            'decimal_hour': time_feats['decimal_hour'],
            'sin_time': time_feats['sin_time'],
            'cos_time': time_feats['cos_time']
        }

        # Enforce exact column ordering from COMMON_FEATURES
        X = pd.DataFrame([feat_dict])[COMMON_FEATURES]

        # 4. Check Out-of-Distribution (OOD) Domain Compatibility
        is_ood, ood_warnings = self.check_out_of_distribution(current_sample)

        # 5. Execute model inference
        predictions = {}
        if self.best_artifact and 'model' in self.best_artifact:
            predictions['best_model'] = float(self.best_artifact['model'].predict(X)[0])
        if self.rf_artifact and 'model' in self.rf_artifact:
            predictions['random_forest'] = float(self.rf_artifact['model'].predict(X)[0])
        if self.xgb_artifact and 'model' in self.xgb_artifact:
            predictions['xgboost'] = float(self.xgb_artifact['model'].predict(X)[0])

        # 6. Update history buffer
        self.history.append(current_sample)

        msg = "Prediction generated successfully."
        if is_ood:
            msg = "Prediction generated with OUT-OF-DISTRIBUTION warning: hardware readings are outside dataset domain."

        return {
            'status': 'READY',
            'message': msg,
            'warmup_count': len(self.history),
            'features': X,
            'feature_dict': feat_dict,
            'predictions': predictions,
            'best_model_name': self.best_artifact.get('model_name') if self.best_artifact else "XGBoost Regressor",
            'is_ood': is_ood,
            'ood_warnings': ood_warnings
        }

    def reset_history(self):
        """Clears historical lag buffer (e.g. on disconnect or port change)."""
        self.history.clear()

if __name__ == "__main__":
    print("=" * 60)
    print("LIVE PREDICTION ENGINE VERIFICATION")
    print("=" * 60)
    engine = LivePredictionEngine()
    print(f"Loaded Best Model: {engine.best_artifact.get('model_name') if engine.best_artifact else 'None'}")
    
    # Test warm-up logic
    s1 = {'temperature': 28.0, 'humidity': 60.0, 'voltage': 32.5, 'current': 7.1, 'power': 230.75}
    r1 = engine.process_sample(s1)
    print(f"Sample 1 -> Status: {r1['status']} | Msg: {r1['message']}")
    assert r1['status'] == 'WARMING_UP', "Warm-up logic failed!"

    s2 = {'temperature': 29.0, 'humidity': 58.0, 'voltage': 33.0, 'current': 7.3, 'power': 240.9}
    r2 = engine.process_sample(s2)
    print(f"Sample 2 -> Status: {r2['status']} | Predictions: {r2['predictions']}")
    assert r2['status'] == 'READY', "Sample 2 prediction failed!"
    print(f"OOD Status: {r2['is_ood']} (Expected False for ~33V / ~7.3A)")

    # Test small educational panel OOD detection
    s3_small = {'temperature': 29.0, 'humidity': 58.0, 'voltage': 6.2, 'current': 0.12, 'power': 0.744}
    r3 = engine.process_sample(s3_small)
    print(f"\nSmall Panel Sample -> OOD Detected: {r3['is_ood']}")
    print(f"OOD Warnings: {r3['ood_warnings']}")
    assert r3['is_ood'] == True, "OOD check failed to flag 6.2V panel!"
    print("\n[+] All LivePredictionEngine verification tests passed successfully!")
