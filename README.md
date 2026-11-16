# Edge-Based Solar Intelligence System

An end-to-end IoT and Machine Learning platform for solar photovoltaic (PV) performance monitoring, time-series feature engineering, and 1-step ahead (~1 hour horizon) solar power forecasting.

---

## 1. Project Introduction
Solar photovoltaic power generation is inherently intermittent due to atmospheric dynamics, cloud passage, and diurnal solar elevation changes. The **Edge-Based Solar Intelligence System** bridges physical edge sensor hardware (ESP32 DevKit V1 with DHT22, LM393 digital LDR, and INA219) with machine learning algorithms (Random Forest and XGBoost) executing on a host computer. By processing historical solar telemetry, computing non-leaking temporal lags, and capturing diurnal solar curves, the system provides accurate 1-step ahead solar power predictions and an interactive dashboard for performance monitoring.

---

## 2. Problem Statement
Small-to-medium solar photovoltaic installations frequently operate without predictive analytics or diagnostic intelligence, leading to undetected panel degradation, inverter mismatch, or sub-optimal battery charge cycles. Furthermore, typical IoT solar projects suffer from:
1. **Target Leakage**: Inadvertently predicting power $P$ using contemporaneous measurements of voltage $V$ and current $I$ (where $P = V \times I$ is an algebraic identity rather than a forecasting task).
2. **Hardware Incompatibility**: Training models on specialized sensors (e.g., continuous pyranometers) that cannot be measured by low-cost edge microcontrollers.
3. **Temporal Incoherency**: Randomly shuffling time-series data during train/test splits, which inflates evaluation metrics through future data leakage.

This project resolves all three issues with a rigorous, leak-free time-series forecasting architecture.

---

## 3. Project Objectives
- **Data Inspection & Provenance**: Audit all daily solar CSV files, normalize schema inconsistencies, and characterize the sampling frequency.
- **Leak-Free Preprocessing**: Construct chronological, within-day lag features and cyclical solar time coordinates.
- **Model Training & Comparison**: Train and tune both a Random Forest Regressor and an XGBoost Regressor using validation-based optimization.
- **Strict Chronological Evaluation**: Evaluate both models on an untouched test partition using MAE, RMSE, and $R^2$ scores.
- **Edge Hardware Integration**: Design ESP32 DevKit V1 firmware that streams structured JSON telemetry matching the Python ML pipeline feature domain.
- **Interactive Web Dashboard**: Provide a Streamlit dashboard with live telemetry monitoring, forecast sandboxes, and visual analytics.
- **Automated Fault Detection**: Provide automated static and dynamic audit scripts to check for dataset, ML, hardware, and software issues.

---

## 4. Hardware Components
1. **Microcontroller**: ESP32 DevKit V1 (30-pin, dual-core Tensilica Xtensa 32-bit LX6 @ 240 MHz).
2. **Temperature & Humidity Sensor**: DHT22 / AM2302 (capacitive humidity sensor and thermistor, calibrated digital signal).
3. **Light Detector**: LM393 Digital LDR Module (potentiometer-adjusted comparator, digital output).
4. **DC Power Monitor**: INA219 (high-side I2C current and voltage sensor with 0.1 $\Omega$ shunt resistor, up to 32V / 2A).
5. **Solar Photovoltaic Panel**: Solar panel module connected across INA219 load circuit terminals.
6. **Host Computer**: PC/Laptop running Python 3.12 with USB serial interface.

---

## 5. Hardware Connections

| Component Pin | ESP32 Pin | Purpose | Electrical Characteristic |
|---|---|---|---|
| **DHT22 DATA** | GPIO 4 | Temperature & Humidity 1-Wire Data | Digital bidir (pull-up resistor) |
| **LM393 DO** | GPIO 34 | Light Level Digital Threshold | Digital input only on ESP32 |
| **INA219 SDA** | GPIO 21 | I2C Serial Data Bus | Digital I2C bus (3.3V logic) |
| **INA219 SCL** | GPIO 22 | I2C Serial Clock Bus | Digital I2C bus (3.3V logic) |
| **VCC** | 3.3V | Regulated DC Supply | 3.3V DC |
| **GND** | GND | Common Circuit Ground | 0V Reference |

> **Crucial Sensor Limitations**:
> - **LDR Module**: Provides a 1-bit digital output (HIGH/LOW). It cannot be treated as a continuous pyranometer irradiance measurement (W/m²).
> - **DHT22**: Measures ambient surrounding air temperature, not photovoltaic cell junction temperature.
> - **INA219**: Measures circuit operating voltage and current under load, distinct from open-circuit voltage ($V_{oc}$) or short-circuit current ($I_{sc}$).

---

## 6. Dataset Description & Source Attribution
The workspace dataset resides in `data/raw/` containing 21 daily CSV files spanning 4 calendar weeks (189 total records):
- **Columns**: `Time`, `Solar Irradiance` (or `SolarIrradiance`), `Temperature`, `Humidity`, `Current`, `Voltage`, `Power`.
- **Measurement Units**:
  - `Solar Irradiance`: incident radiant flux in Watts per square meter ($\text{W/m}^2$).
  - `Temperature`: degrees Celsius ($^\circ\text{C}$).
  - `Humidity`: relative atmospheric humidity percent ($\%$).
  - `Current`: direct current in Amperes ($\text{A}$).
  - `Voltage`: direct voltage in Volts ($\text{V}$).
  - `Power`: active electrical power output in Watts ($\text{W}$).
- **Sampling Interval**: Average of 59.4 minutes between successive measurements (~9 daylight observations per day from ~08:00 AM to ~16:30 PM).
- **Physical Law Verification**: The physical law $P = V \times I$ holds strictly across all observations within rounding precision ($< 0.05\,\text{W}$).

---

## 7. Data Preprocessing & Feature Engineering
Located in `src/preprocessing.py`:
1. **Schema Normalization**: Strips whitespace and standardizes column headers across all raw files.
2. **Handling Day 28**: Uses `solar_data_tue_28th.csv` as the canonical day 28 series to maintain a clean monotonic sequence while preserving both raw files.
3. **Cyclical Solar Coordinates**:
   $$\sin\_time = \sin\left(\frac{2\pi \cdot \text{decimal\_hour}}{24}\right), \quad \cos\_time = \cos\left(\frac{2\pi \cdot \text{decimal\_hour}}{24}\right)$$
4. **Within-Day Lag Features**:
   - `power_lag_1` ($P_{t-1}$)
   - `voltage_lag_1` ($V_{t-1}$)
   - `current_lag_1` ($I_{t-1}$)
   - `temp_lag_1` ($T_{t-1}$)
   - `humidity_lag_1` ($H_{t-1}$)
   - `power_trend` ($P_t - P_{t-1}$)
   *Lags are grouped by day to prevent artificial transitions across midnight.*
5. **Target Variable**:
   $$\text{Target} = P_{t+1} \quad \text{(Next daylight observation, $\sim$1-hour forecast horizon)}$$
6. **Chronological Splitting**:
   - **Training Set**: First 14 days (Days 6–23) $\rightarrow$ 98 samples (70%)
   - **Validation Set**: Next 3 days (Days 24, 27, 28) $\rightarrow$ 21 samples (15%)
   - **Test Set**: Final 3 days (Days 29, 30, 31) $\rightarrow$ 21 samples (15%)
   *Strictly chronological: no random shuffling, preventing future data leakage.*

---

## 8. Machine Learning Methodology
To guarantee that the machine learning models can be executed using real telemetry from the ESP32 hardware:
- We exclude pyranometer solar irradiance ($\text{W/m}^2$) from the input feature set because the ESP32 only has a 1-bit digital LDR module.
- We exclude contemporaneous voltage and current from predicting contemporaneous power (preventing trivial $V \times I$ leakage).
- All 14 input features (`Temperature`, `Humidity`, `Voltage`, `Current`, `Power`, `power_lag_1`, `voltage_lag_1`, `current_lag_1`, `temp_lag_1`, `humidity_lag_1`, `power_trend`, `decimal_hour`, `sin_time`, `cos_time`) are derived from the physical sensors or the host clock.

---

## 9. Random Forest Explanation
The Random Forest Regressor combines an ensemble of decorrelated decision trees built via bootstrap aggregating (bagging):
- **Hyperparameter Selection**: Tuned over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` on the validation partition.
- **Optimal Hyperparameters**: `n_estimators=50`, `max_depth=4`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Top Predictive Features**: `cos_time` (49.99%), `voltage_lag_1` (17.68%), `Voltage` (10.88%), `power_trend` (5.71%).

---

## 10. XGBoost Explanation
The XGBoost Regressor fits decision trees sequentially by minimizing a regularized objective function using first- and second-order gradient approximations:
- **Hyperparameter Selection**: Tuned over `n_estimators`, `max_depth`, `learning_rate`, and `subsample` on the validation partition.
- **Optimal Hyperparameters**: `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`.
- **Top Predictive Features**: `cos_time` (50.38%), `voltage_lag_1` (13.19%), `Voltage` (8.79%), `power_trend` (5.43%), `power_lag_1` (5.07%).

---

## 11. Model Evaluation & Comparison

Both models were evaluated on the **untouched chronological test set** (Days 29, 30, 31):

| Metric | Random Forest Regressor | XGBoost Regressor | Optimal Model |
|---|---|---|---|
| **Validation MAE** | 9.360 W | **8.069 W** | **XGBoost (-1.29 W)** |
| **Validation RMSE** | 11.392 W | **10.755 W** | **XGBoost (-0.64 W)** |
| **Validation $R^2$** | 0.6475 | **0.6858** | **XGBoost (+0.038)** |
| **Test Set MAE** | 8.979 W | **6.473 W** | **XGBoost (-2.51 W)** |
| **Test Set RMSE** | 10.185 W | **7.657 W** | **XGBoost (-2.53 W)** |
| **Test Set $R^2$** | 0.1679 | **0.5297** | **XGBoost (+0.362)** |

### Scientific Justification
XGBoost outperforms Random Forest across every metric on both validation and test sets:
- **Gradient Optimization**: Stepwise shrinkage (`learning_rate=0.05`) allows XGBoost to approximate subtle diurnal solar changes without aggressive step variance.
- **L1/L2 Regularization**: Built-in tree regularization avoids overfitting to specific daily microclimates, yielding superior generalization on test days.

---

## 12. ESP32 Integration & Live Inference
- **Firmware**: Located in `esp32/solar_sensor.ino`. Formats telemetry into pure JSON packets over USB serial at 115200 baud:
  ```json
  {"temperature":32.5,"humidity":60.2,"ldr":1,"voltage":6.20,"current_ma":120.5,"power_mw":747.1}
  ```
- **Inference Location**: Model inference runs host-side in Python via `src/serial_reader.py` and `src/live_prediction.py`.
- **Hardware Disconnected Safety**: If the ESP32 is not plugged in, the system explicitly marks hardware testing as pending. No fabricated sensor readings are emitted.

---

## 13. Interactive Dashboard
The Streamlit dashboard (`dashboard.py`) includes 5 modular views:
1. **Overview & Live Telemetry**: Live sensor gauges, COM port selector, and hardware status.
2. **ML Forecasting Engine**: Interactive test sandbox allowing users to pick chronological test samples and compare RF, XGBoost, and ground truth predictions.
3. **Model Performance & Metrics**: Quantitative scorecards, comparison tables, and leakage prevention notes.
4. **Visual Analytics**: Interactive tabs showing all 9 generated Matplotlib/Seaborn figures.
5. **System Diagnostics**: Hardware detection status, model loading checks, dataset counts, and the automated fault audit log.

---

## 14. Installation & Setup

1. **Activate the Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 15. How to Run the Project

- **Run Full Pipeline End-to-End**:
  ```powershell
  python main.py --all
  ```
- **Run Individual Modules**:
  ```powershell
  python src/data_inspection.py    # Inspect dataset & produce outputs/dataset_report.txt
  python src/preprocessing.py      # Build lag features & split partitions
  python src/train_random_forest.py # Train & tune Random Forest
  python src/train_xgboost.py       # Train & tune XGBoost
  python src/evaluate_models.py     # Evaluate on test set & render graphs
  python src/verify_models.py       # Deserialization & prediction sanity check
  python src/fault_checker.py       # Automated fault detection
  ```
- **Launch Interactive Dashboard**:
  ```powershell
  streamlit run dashboard.py
  ```

---

## 16. Results & Limitations
- **Results**: The system reliably achieves a test MAE of 6.47 W (~2.4% relative error on a ~270W peak panel) for a 1-step ahead (~1-hour) solar power forecast.
- **Limitations**:
  - The dataset sampling frequency is ~1 hour; forecasting cannot claim a 10-second horizon.
  - The hardware LM393 LDR sensor provides only a 1-bit threshold rather than continuous irradiance.
  - Hardware testing requires the physical ESP32 to be connected via USB.

---

## 17. Future Improvements
1. **MicroPython / TinyML Deployment**: Quantize the XGBoost tree ensemble into C headers using `m2cgen` or `tinyml` to run lightweight inference directly on the ESP32 MCU.
2. **Continuous Analog Irradiance Sensor**: Replace the 1-bit LM393 module with an analog solar pyranometer or calibrated photodiode on an analog ADC pin.
3. **NTP Time Synchronization**: Connect the ESP32 to local Wi-Fi to fetch UTC network timestamps directly via SNTP.

