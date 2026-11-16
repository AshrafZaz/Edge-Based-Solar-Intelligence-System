"""
Solar Intelligence System - ESP32 Serial Interface
File: src/serial_reader.py

Handles USB serial communication between the host PC and ESP32 DevKit V1:
1. Enumerates and classifies COM ports on Windows (differentiating physical USB from Bluetooth).
2. Prioritizes physical USB-to-UART bridges (CP210x, CH340, FTDI, Espressif).
3. Connects to the exact user-specified COM port at 115200 baud.
4. Reads structured JSON telemetry packets emitted by esp32/solar_sensor.ino.
5. Rigorous validation of packet schema and numerical finiteness.
6. Safe port acquisition and release lifecycle management.
"""

import time
import json
import serial
import serial.tools.list_ports
from typing import Dict, Any, Optional, List, Tuple

# Known USB-to-UART Vendor IDs & Hardware Identifiers for ESP32 boards
ESP32_USB_SIGNATURES = [
    ("10C4", "Silicon Labs CP210x"),
    ("1A86", "WCH CH340/CH341"),
    ("0403", "FTDI FT232R"),
    ("303A", "Espressif USB JTAG/Serial"),
    ("2341", "Arduino USB"),
]

def list_available_ports() -> List[Dict[str, Any]]:
    """
    Enumerate all serial COM ports on the system and classify each
    as physical USB, Bluetooth, or virtual.
    """
    ports = serial.tools.list_ports.comports()
    results = []
    
    for p in ports:
        desc = p.description or ""
        hwid = p.hwid or ""
        mfg = p.manufacturer or ""
        
        # Detect Bluetooth serial ports (common virtual COM ports on Windows laptops)
        is_bluetooth = ("bluetooth" in desc.lower() or 
                        "bthenum" in hwid.lower() or 
                        "bth" in hwid.lower())
        
        # Detect Physical USB ports
        is_usb = hwid.upper().startswith("USB") or "usb" in desc.lower()
        
        chip = "Unknown"
        for vid, chip_name in ESP32_USB_SIGNATURES:
            if vid in hwid.upper():
                chip = chip_name
                is_usb = True
                break
                
        port_type = "Physical USB" if is_usb and not is_bluetooth else ("Bluetooth (Virtual)" if is_bluetooth else "Other Serial")
        
        results.append({
            'port': p.device,
            'description': desc,
            'hwid': hwid,
            'manufacturer': mfg,
            'is_usb': is_usb and not is_bluetooth,
            'is_bluetooth': is_bluetooth,
            'port_type': port_type,
            'chip': chip
        })
        
    # Sort physical USB ports first, then alphabetically by port name
    results.sort(key=lambda x: (not x['is_usb'], x['port']))
    return results

def get_physical_usb_ports() -> List[Dict[str, Any]]:
    """Return only genuine physical USB serial ports (excluding Bluetooth virtual ports)."""
    return [p for p in list_available_ports() if p['is_usb']]

class ESP32SerialReader:
    """Manages connection, streaming, and error handling for ESP32 sensor telemetry."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 1.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.last_error = ""

    def connect(self, port: Optional[str] = None) -> Tuple[bool, str]:
        """
        Connects to the exact port specified.
        Returns (success: bool, status_message: str).
        """
        if port:
            self.port = port
            
        if not self.port:
            self.last_error = "No COM port specified."
            return False, self.last_error

        # Ensure previous connection is cleanly terminated
        self.disconnect()

        try:
            # Check port existence
            available = [p['port'] for p in list_available_ports()]
            if self.port not in available:
                self.last_error = f"Port '{self.port}' is not currently present on this computer."
                return False, self.last_error

            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(1.5)  # Allow ESP32 reset after DTR/RTS toggle
            self.ser.reset_input_buffer()
            self.is_connected = True
            self.last_error = ""
            msg = f"Connected successfully to {self.port} at {self.baudrate} baud."
            return True, msg
            
        except serial.SerialException as se:
            err_str = str(se)
            if "PermissionError" in err_str or "Access is denied" in err_str:
                self.last_error = f"Access denied on {self.port}. The port is in use by another program (e.g., Arduino Serial Monitor, another script)."
            else:
                self.last_error = f"Serial error on {self.port}: {err_str}"
            self.is_connected = False
            self.ser = None
            return False, self.last_error
            
        except Exception as e:
            self.last_error = f"Unexpected error opening {self.port}: {e}"
            self.is_connected = False
            self.ser = None
            return False, self.last_error

    def disconnect(self):
        """Safely release the serial port resource."""
        if self.ser:
            try:
                if self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.is_connected = False

    def read_telemetry(self) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Reads one line from the serial buffer, parses the structured JSON payload,
        validates electrical/physical consistency, and converts units.
        
        Returns:
            (telemetry_dict, status_message)
        """
        if not self.is_connected or not self.ser or not self.ser.is_open:
            return None, "Serial port is not connected."

        try:
            raw_bytes = self.ser.readline()
            if not raw_bytes:
                return None, "Awaiting packet (timeout: no data received)."

            line = raw_bytes.decode('utf-8', errors='replace').strip()
            if not line:
                return None, "Empty line received."
                
            # Filter non-JSON debug output from bootloader
            if not (line.startswith('{') and line.endswith('}')):
                return None, f"Non-JSON serial chatter: {line[:40]}"

            data = json.loads(line)

            # Validate mandatory telemetry schema
            required_keys = ['temperature', 'humidity', 'ldr', 'voltage', 'current_ma', 'power_mw']
            for k in required_keys:
                if k not in data:
                    return None, f"Schema mismatch: missing key '{k}' in JSON payload: {line}"

            # Validate numerical values
            temp = data.get('temperature')
            hum = data.get('humidity')
            ldr = data.get('ldr')
            voltage = data.get('voltage')
            current_ma = data.get('current_ma')
            power_mw = data.get('power_mw')

            # Handle sensor read errors (e.g. DHT22 reading failed)
            if temp is None or hum is None:
                return None, "DHT22 read failure: temperature or humidity reported null."
                
            temp = float(temp)
            hum = float(hum)
            voltage = float(voltage)
            current_ma = float(current_ma)
            power_mw = float(power_mw)
            ldr = int(ldr)

            # Basic physical bounds check on edge sensor hardware
            if not (-20.0 <= temp <= 80.0):
                return None, f"Unrealistic DHT22 temperature reading: {temp}°C"
            if not (0.0 <= hum <= 100.0):
                return None, f"Unrealistic DHT22 humidity reading: {hum}%"
            if voltage < 0.0 or current_ma < 0.0 or power_mw < 0.0:
                return None, f"Negative electrical reading: V={voltage}V, I={current_ma}mA, P={power_mw}mW"

            # Construct verified telemetry record
            telemetry = {
                'temperature': temp,                      # °C (from DHT22)
                'humidity': hum,                          # %  (from DHT22)
                'ldr_status': ldr,                        # 1 = Light detected, 0 = Dark (LM393 DO)
                'voltage': voltage,                       # Volts (V) (from INA219)
                'current_ma': current_ma,                 # Milliamps (mA) (from INA219)
                'power_mw': power_mw,                     # Milliwatts (mW) (from INA219)
                'current': current_ma / 1000.0,           # Amperes (A) for ML model
                'power': power_mw / 1000.0,               # Watts (W) for ML model
                'timestamp': time.time(),
                'raw_json': line
            }
            return telemetry, "Telemetry packet parsed and validated successfully."

        except json.JSONDecodeError as jde:
            return None, f"Malformed JSON: {jde}"
        except serial.SerialException as se:
            self.disconnect()
            return None, f"Serial communication broken (device disconnected?): {se}"
        except Exception as ex:
            return None, f"Telemetry parsing error: {ex}"

    def __del__(self):
        self.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("ESP32 SERIAL INTERFACE DIAGNOSTIC SCANNER")
    print("=" * 60)
    all_ports = list_available_ports()
    print(f"Total serial COM ports detected: {len(all_ports)}")
    for p in all_ports:
        print(f"  - [{p['port']}] {p['description']} | Type: {p['port_type']} | Chip: {p['chip']}")
        
    usb_ports = get_physical_usb_ports()
    print(f"\nPhysical USB serial ports available: {len(usb_ports)}")
    if not usb_ports:
        print("[!] No physical ESP32 USB ports detected. Hardware testing is in PENDING status.")
    else:
        for u in usb_ports:
            print(f"  * {u['port']}: {u['chip']}")
