/*
  Edge-Based Solar Intelligence System - ESP32 Firmware
  File: esp32/solar_sensor.ino
  
  Target Board: ESP32 DevKit V1
  Sensors:
    - DHT22 (Ambient Temperature & Humidity) -> GPIO 4
    - LM393 Digital LDR Module (Light Status) -> GPIO 34 (Input-only pin on ESP32)
    - INA219 (I2C Voltage, Current, Power Sensor) -> SDA: GPIO 21, SCL: GPIO 22
    - Solar Panel connected across INA219 Vin+ and Vin- load circuit
    - VCC -> 3.3V
    - GND -> GND

  Baud Rate: 115200
  Output Format: Pure JSON string per line
  Example:
  {"temperature":32.5,"humidity":60.2,"ldr":1,"voltage":6.20,"current_ma":120.5,"power_mw":747.1}

  CRITICAL HARDWARE & ELECTRICAL SPECIFICATIONS:
    1. INA219 Sensor Breakout:
       - Max Bus Voltage: 32.0V DC (chip register limit).
       - Max Shunt Current: 3.2A with standard 0.1 ohm shunt resistor (320mV / 0.1 ohm).
       - Max Measurable Power: ~102.4W.
    2. Educational Lab Panel vs Dataset Range:
       - Small desktop lab panels typically produce 5V to 18V and 50mA to 1.2A (<20W).
       - The historical dataset was recorded from a 30-37V, 6-8A (~270W peak) PV array.
       - Operating a small lab panel with INA219 is electrically safe, but measurements
         fall below the dataset domain. The host-side ML pipeline detects this and flags
         an Out-of-Distribution (OOD) warning.
    3. LM393 LDR Sensor:
       - Digital output DO provides a 1-bit comparator threshold (HIGH/LOW).
       - It is NOT a calibrated pyranometer (W/m²). Do not treat digital status as irradiance.

  Libraries Required:
    - DHT sensor library by Adafruit (or DHTesp)
    - Adafruit Unified Sensor
    - Adafruit INA219
    - Wire (built-in)
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <Adafruit_INA219.h>

// Pin Definitions
#define DHTPIN 4
#define DHTTYPE DHT22
#define LDRPIN 34       // GPIO 34 is input-only on ESP32, ideal for digital DO from LM393
#define I2C_SDA 21
#define I2C_SCL 22

// Sensor Instances
DHT dht(DHTPIN, DHTTYPE);
Adafruit_INA219 ina219;

bool ina219_connected = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Wait for serial monitor to connect (up to 3 seconds)
  }

  // Initialize GPIO
  pinMode(LDRPIN, INPUT);

  // Initialize I2C with explicit ESP32 pins
  Wire.begin(I2C_SDA, I2C_SCL);

  // Initialize DHT22
  dht.begin();

  // Initialize INA219
  if (ina219.begin()) {
    ina219_connected = true;
    // By default, INA219 measures up to 32V, 2A with 0.1 ohm shunt
    // Can also set calibration if measuring smaller panel:
    // ina219.setCalibration_32V_1A();
  } else {
    ina219_connected = false;
  }
}

void loop() {
  // DHT22 sampling cycle minimum is 2 seconds
  delay(2000);

  // 1. Read DHT22
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // Celsius

  // Fallback checks for DHT sensor read failure
  bool dht_valid = !isnan(temperature) && !isnan(humidity);
  if (!dht_valid) {
    temperature = -999.0;
    humidity = -999.0;
  }

  // 2. Read Digital LDR Module
  // LM393 comparator outputs LOW when light intensity exceeds potentiometer threshold,
  // HIGH when dark. Reading raw digital state:
  int ldr_raw = digitalRead(LDRPIN);
  // Normalize: 1 = Light detected, 0 = Dark (or pass raw state directly)
  int ldr_status = (ldr_raw == LOW) ? 1 : 0;

  // 3. Read INA219 Voltage, Current, Power
  float bus_voltage_v = 0.0;
  float current_ma = 0.0;
  float power_mw = 0.0;

  if (ina219_connected) {
    float shuntvoltage_mv = ina219.getShuntVoltage_mV();
    float busvoltage_v = ina219.getBusVoltage_V();
    bus_voltage_v = busvoltage_v + (shuntvoltage_mv / 1000.0);
    current_ma = ina219.getCurrent_mA();
    power_mw = ina219.getPower_mW();

    // Prevent negative readings from micro-offsets when panel is disconnected
    if (bus_voltage_v < 0.0) bus_voltage_v = 0.0;
    if (current_ma < 0.0) current_ma = 0.0;
    if (power_mw < 0.0) power_mw = 0.0;
  } else {
    // Attempt re-initialization if INA219 disconnected
    if (ina219.begin()) {
      ina219_connected = true;
    }
  }

  // Construct structured JSON output
  // Example: {"temperature":32.5,"humidity":60.2,"ldr":1,"voltage":6.20,"current_ma":120.5,"power_mw":747.1}
  Serial.print("{\"temperature\":");
  if (dht_valid) {
    Serial.print(temperature, 1);
  } else {
    Serial.print("null");
  }
  
  Serial.print(",\"humidity\":");
  if (dht_valid) {
    Serial.print(humidity, 1);
  } else {
    Serial.print("null");
  }
  
  Serial.print(",\"ldr\":");
  Serial.print(ldr_status);
  
  Serial.print(",\"voltage\":");
  Serial.print(bus_voltage_v, 2);
  
  Serial.print(",\"current_ma\":");
  Serial.print(current_ma, 1);
  
  Serial.print(",\"power_mw\":");
  Serial.print(power_mw, 1);
  
  Serial.println("}");
}

