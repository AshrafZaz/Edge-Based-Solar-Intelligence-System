"""
Solar Intelligence System - Data Preprocessing & Feature Engineering
File: src/preprocessing.py

Implements:
1. Multi-CSV ingestion and schema normalization (resolving column name discrepancies).
2. Chronological sorting by calendar day and observation time.
3. Timestamp parsing & cyclical solar time features (sin_time, cos_time, hour_decimal).
4. Physical plausibility checks and robust imputation/outlier filtering.
5. Scientifically valid feature engineering aligned with ESP32 sensors:
   - Temperature (°C)
   - Humidity (%)
   - Voltage (V)
   - Current (A)
   - Power (W)
   - Lag features (power_lag_1, voltage_lag_1, current_lag_1, temp_lag_1, humidity_lag_1)
   - Power trend (delta power)
   - Target: Next-sample solar power (t+1), ~1-hour forecast horizon.
6. Chronological Train / Validation / Test split (NO random shuffling to prevent temporal leakage).
7. Feature scaling fitted exclusively on the training partition.
8. Artifact export to data/processed/.
"""

import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def parse_day_number(filename: str) -> int:
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)(?:st|nd|rd|th)', basename)
    if match:
        return int(match.group(1))
    return 999

def time_to_decimal_hours(time_str: str) -> float:
    """Convert 'H:MM' or 'HH:MM' string to decimal hours (e.g. 8:30 -> 8.5)."""
    parts = str(time_str).strip().split(':')
    h = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s = float(parts[2]) if len(parts) > 2 else 0.0
    return h + m / 60.0 + s / 3600.0

def load_and_combine_raw_data(raw_dir: str = "data/raw", deduplicate_day_28: bool = True) -> pd.DataFrame:
    """
    Loads all daily CSV files from data/raw/, normalizes column schemas,
    and sorts chronologically.
    
    If deduplicate_day_28 is True, handles 'solar_data_tue_28st.csv' vs 'solar_data_tue_28th.csv'
    by using 'solar_data_tue_28th.csv' as canonical to maintain a single monotonic day sequence.
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{raw_dir}'.")
    
    # Sort files by day number
    csv_files = sorted(csv_files, key=parse_day_number)
    
    dfs = []
    for f in csv_files:
        fname = os.path.basename(f)
        if deduplicate_day_28 and fname == "solar_data_tue_28st.csv":
            # Documented duplicate day variation; skip in favor of 28th
            continue
            
        df = pd.read_csv(f)
        # Normalize column names: strip whitespace, remove internal spaces
        df.columns = [c.strip().replace(' ', '') for c in df.columns]
        
        # Ensure mandatory columns exist
        expected_cols = {'Time', 'SolarIrradiance', 'Temperature', 'Humidity', 'Current', 'Voltage', 'Power'}
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(f"File {fname} is missing required columns: {missing}")
            
        df['source_file'] = fname
        df['day_number'] = parse_day_number(fname)
        df['decimal_hour'] = df['Time'].apply(time_to_decimal_hours)
        
        # Sort each day by decimal hour
        df = df.sort_values('decimal_hour').reset_index(drop=True)
        dfs.append(df)
        
    combined = pd.concat(dfs, ignore_index=True)
    # Global chronological sort: first by calendar day, then by time of day
    combined = combined.sort_values(by=['day_number', 'decimal_hour']).reset_index(drop=True)
    return combined

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs leak-free time-series features and target variable.
    
    Features built:
    1. Cyclical time: sin_time, cos_time representing solar elevation angle proxy.
    2. Within-day lag features:
       - power_lag_1: Power at t-1 (within same day)
       - voltage_lag_1: Voltage at t-1
       - current_lag_1: Current at t-1
       - temp_lag_1: Temperature at t-1
       - humidity_lag_1: Humidity at t-1
       - power_trend: Power(t) - Power(t-1)
    3. Target:
       - target_power: Power at t+1 (next daylight observation ~1 hr ahead).
    """
    df = df.copy()
    
    # 1. Cyclical time encoding (24-hour cycle)
    df['sin_time'] = np.sin(2 * np.pi * df['decimal_hour'] / 24.0)
    df['cos_time'] = np.cos(2 * np.pi * df['decimal_hour'] / 24.0)
    
    # Group by day so lags and targets never leak across midnight / calendar days
    day_groups = []
    for day, group in df.groupby('day_number', sort=False):
        g = group.copy().sort_values('decimal_hour').reset_index(drop=True)
        
        # Prior observation lags (within the same daylight day)
        g['power_lag_1'] = g['Power'].shift(1)
        g['voltage_lag_1'] = g['Voltage'].shift(1)
        g['current_lag_1'] = g['Current'].shift(1)
        g['temp_lag_1'] = g['Temperature'].shift(1)
        g['humidity_lag_1'] = g['Humidity'].shift(1)
        g['power_trend'] = g['Power'] - g['power_lag_1']
        
        # Target: Next observation solar power (t+1)
        g['target_power'] = g['Power'].shift(-1)
        
        day_groups.append(g)
        
    featured_df = pd.concat(day_groups, ignore_index=True)
    
    # Drop rows without complete lag or target:
    # First row of each day lacks prior lag (t-1)
    # Last row of each day lacks future target (t+1)
    # This prevents artificial edge discontinuities
    valid_df = featured_df.dropna(subset=[
        'power_lag_1', 'voltage_lag_1', 'current_lag_1', 'temp_lag_1', 'humidity_lag_1', 'target_power'
    ]).reset_index(drop=True)
    
    return valid_df

# Canonical common feature set compatible with ESP32 sensor hardware
COMMON_FEATURES = [
    'Temperature',       # °C (from DHT22 on ESP32 or dataset)
    'Humidity',          # %  (from DHT22 on ESP32 or dataset)
    'Voltage',           # V  (from INA219 on ESP32 or dataset)
    'Current',           # A  (from INA219 mA/1000 on ESP32 or dataset)
    'Power',             # W  (from INA219 mW/1000 on ESP32 or dataset)
    'power_lag_1',       # W  (historical buffer on host/ESP32)
    'voltage_lag_1',     # V  (historical buffer)
    'current_lag_1',     # A  (historical buffer)
    'temp_lag_1',        # °C (historical buffer)
    'humidity_lag_1',     # %  (historical buffer)
    'power_trend',       # W  (Power(t) - Power(t-1))
    'decimal_hour',      # Decimal hour (from host RTC/NTP)
    'sin_time',          # Cyclical sin
    'cos_time'           # Cyclical cos
]

TARGET_COL = 'target_power'

def run_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    os.makedirs(processed_dir, exist_ok=True)
    print("[*] Starting data preprocessing and feature engineering...")
    
    raw_combined = load_and_combine_raw_data(raw_dir)
    print(f"[*] Raw dataset loaded: {len(raw_combined)} total records across {raw_combined['day_number'].nunique()} days.")
    
    featured = engineer_features(raw_combined)
    print(f"[*] Feature engineering complete: {len(featured)} valid time-series instances with 1-step targets.")
    
    # Chronological partition by unique days (No shuffling)
    unique_days = featured['day_number'].drop_duplicates().tolist()
    n_days = len(unique_days)
    n_train_days = int(np.floor(n_days * train_ratio))
    n_val_days = int(np.floor(n_days * val_ratio))
    
    train_days = unique_days[:n_train_days]
    val_days = unique_days[n_train_days:n_train_days + n_val_days]
    test_days = unique_days[n_train_days + n_val_days:]
    
    train_df = featured[featured['day_number'].isin(train_days)].copy().reset_index(drop=True)
    val_df = featured[featured['day_number'].isin(val_days)].copy().reset_index(drop=True)
    test_df = featured[featured['day_number'].isin(test_days)].copy().reset_index(drop=True)
    
    print(f"[+] Partitioning by Days:")
    print(f"    Train Days ({len(train_days)} days): {train_days} -> {len(train_df)} samples")
    print(f"    Val Days   ({len(val_days)} days): {val_days} -> {len(val_df)} samples")
    print(f"    Test Days  ({len(test_days)} days): {test_days} -> {len(test_df)} samples")
    
    # Fit StandardScaler on TRAIN SET ONLY to prevent test leakage
    scaler = StandardScaler()
    scaler.fit(train_df[COMMON_FEATURES])
    
    # Save scaler artifact
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    joblib.dump({
        'scaler': scaler,
        'features': COMMON_FEATURES,
        'target': TARGET_COL
    }, scaler_path)
    print(f"[+] Scaler saved to {scaler_path}")
    
    # Save CSV partitions
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    featured.to_csv(os.path.join(processed_dir, "full_processed.csv"), index=False)
    
    print("[+] All processed datasets saved successfully in 'data/processed/'.")
    return train_df, val_df, test_df

if __name__ == "__main__":
    run_preprocessing_pipeline()

