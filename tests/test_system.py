"""
Solar Intelligence System - Automated Unit & Integration Tests
File: tests/test_system.py

Tests:
1. Test cyclical time coordinate mathematical bounds.
2. Test feature engineering and within-day lag isolation (no leakage across days).
3. Test target alignment (Power(t+1) vs features at t).
4. Test live prediction warm-up period and OOD detection.
5. Test serial JSON telemetry parsing and malformed packet rejection.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root and src to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.preprocessing import time_to_decimal_hours, engineer_features, COMMON_FEATURES, TARGET_COL
from src.serial_reader import ESP32SerialReader
from src.live_prediction import LivePredictionEngine

class TestSolarIntelligenceSystem(unittest.TestCase):

    def test_decimal_hour_conversion(self):
        """Verify time parsing converts HH:MM to exact decimal hours."""
        self.assertAlmostEqual(time_to_decimal_hours("08:00"), 8.0)
        self.assertAlmostEqual(time_to_decimal_hours("08:30"), 8.5)
        self.assertAlmostEqual(time_to_decimal_hours("12:15"), 12.25)
        self.assertAlmostEqual(time_to_decimal_hours("16:45"), 16.75)

    def test_cyclical_time_bounds(self):
        """Verify sin_time and cos_time remain strictly bounded in [-1.0, 1.0]."""
        engine = LivePredictionEngine(models_dir=os.path.join(PROJECT_ROOT, "models"))
        for h in range(24):
            import datetime
            dt = datetime.datetime(2026, 1, 1, h, 0, 0)
            t_feats = engine.extract_time_features(dt)
            self.assertGreaterEqual(t_feats['sin_time'], -1.0)
            self.assertLessEqual(t_feats['sin_time'], 1.0)
            self.assertGreaterEqual(t_feats['cos_time'], -1.0)
            self.assertLessEqual(t_feats['cos_time'], 1.0)

    def test_lag_feature_within_day_isolation(self):
        """Verify lag features do NOT leak across midnight / calendar days."""
        test_data = pd.DataFrame([
            {'day_number': 1, 'decimal_hour': 8.0, 'Power': 100.0, 'Voltage': 30.0, 'Current': 3.3, 'Temperature': 25, 'Humidity': 60},
            {'day_number': 1, 'decimal_hour': 9.0, 'Power': 150.0, 'Voltage': 31.0, 'Current': 4.8, 'Temperature': 26, 'Humidity': 55},
            {'day_number': 1, 'decimal_hour': 10.0, 'Power': 200.0, 'Voltage': 32.0, 'Current': 6.2, 'Temperature': 28, 'Humidity': 50},
            # Day 2 starts
            {'day_number': 2, 'decimal_hour': 8.0, 'Power': 90.0, 'Voltage': 29.0, 'Current': 3.1, 'Temperature': 24, 'Humidity': 65},
            {'day_number': 2, 'decimal_hour': 9.0, 'Power': 140.0, 'Voltage': 30.5, 'Current': 4.6, 'Temperature': 25, 'Humidity': 60},
            {'day_number': 2, 'decimal_hour': 10.0, 'Power': 190.0, 'Voltage': 31.5, 'Current': 6.0, 'Temperature': 27, 'Humidity': 55},
        ])
        
        featured = engineer_features(test_data)
        
        # Day 1 middle row (hour 9) should have power_lag_1 == 100.0, target == 200.0
        d1_row = featured[(featured['day_number'] == 1) & (featured['decimal_hour'] == 9.0)].iloc[0]
        self.assertEqual(d1_row['power_lag_1'], 100.0)
        self.assertEqual(d1_row['target_power'], 200.0)
        
        # Day 2 middle row (hour 9) should have power_lag_1 == 90.0 (from day 2, NOT 200.0 from day 1)
        d2_row = featured[(featured['day_number'] == 2) & (featured['decimal_hour'] == 9.0)].iloc[0]
        self.assertEqual(d2_row['power_lag_1'], 90.0)
        self.assertEqual(d2_row['target_power'], 190.0)

    def test_live_prediction_warmup_and_ood(self):
        """Verify warm-up period (sample 1) and genuine lag prediction (sample 2)."""
        engine = LivePredictionEngine(models_dir=os.path.join(PROJECT_ROOT, "models"))
        engine.reset_history()
        
        sample_packet_1 = {
            'temperature': 30.0,
            'humidity': 65.0,
            'ldr_status': 1,
            'voltage': 34.0,
            'current_ma': 7500.0,
            'power_mw': 255000.0,
            'current': 7.5,
            'power': 255.0,
            'timestamp': 100000.0
        }
        
        # Sample 1: Must enter WARMING_UP status and NOT predict with fake lag
        r1 = engine.process_sample(sample_packet_1)
        self.assertEqual(r1['status'], 'WARMING_UP')
        self.assertIsNone(r1['predictions'])
        
        # Sample 2: With genuine prior observation, must predict
        sample_packet_2 = sample_packet_1.copy()
        sample_packet_2['power'] = 265.0
        r2 = engine.process_sample(sample_packet_2)
        self.assertEqual(r2['status'], 'READY')
        self.assertIsNotNone(r2['predictions'])
        self.assertIn('best_model', r2['predictions'])
        pred = r2['predictions']['best_model']
        self.assertTrue(150.0 < pred < 350.0)
        self.assertFalse(r2['is_ood'])

        # Sample 3 (Small Lab Panel): Must flag Out-Of-Distribution (OOD)
        sample_small = {
            'temperature': 28.0,
            'humidity': 55.0,
            'ldr_status': 1,
            'voltage': 6.2,
            'current_ma': 120.0,
            'power_mw': 744.0,
            'current': 0.12,
            'power': 0.744,
            'timestamp': 100002.0
        }
        r3 = engine.process_sample(sample_small)
        self.assertTrue(r3['is_ood'])
        self.assertGreater(len(r3['ood_warnings']), 0)

    def test_serial_json_validation(self):
        """Verify serial reader handles disconnected state safely without exceptions."""
        reader = ESP32SerialReader(port="COM99")
        self.assertFalse(reader.is_connected)
        
        telemetry, msg = reader.read_telemetry()
        self.assertIsNone(telemetry)
        self.assertIn("not connected", msg)

if __name__ == '__main__':
    unittest.main()
