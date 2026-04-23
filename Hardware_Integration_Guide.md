# PRANA One: Hardware Integration Guide

This document outlines how to transition from the current **Simulation Phase** to **Real-World Hardware Integration**.

## 1. Hardware Requirements
*   **Microcontroller**: ESP32 or Arduino Nano 33 IoT (must support Bluetooth/WiFi).
*   **IMU Sensor**: MPU6050 (6-axis: Accelerometer + Gyroscope).
*   **Connectivity**: Bluetooth Low Energy (BLE) is recommended for mobile app pairing.

## 2. Sensor Data Specs
The AI model and rule-based logic expect the following units:
*   **Accelerometer (ax, ay, az)**: m/s² (Scale ±4g or ±8g).
*   **Gyroscope (gx, gy, gz)**: rad/s (Scale ±1000°/s or ±2000°/s).

## 3. Integration Points

### A. Mobile App (Flutter/Web)
In `app/index.html`, the `triggerCrash()` function is currently called by a demo button.
*   **Action**: Implement a Bluetooth listener.
*   **Logic**:
    ```javascript
    // Pseudo-code for Bluetooth listener
    device.characteristic.subscribe((data) => {
        let sensorValues = parseIMU(data);
        // Call the rule-based logic from finalize.py
        if (detectActivity(sensorValues)) {
            triggerCrash(); // Triggers the SOS flow
        }
    });
    ```

### B. Python Backend (Edge/Cloud)
In `backend/inference.py`, replace `simulate_sensor_window()` with a real Serial/Socket reader.
*   **Action**:
    ```python
    import serial
    ser = serial.Serial('COM3', 115200)
    
    def get_real_sensor_data():
        line = ser.readline().decode('utf-8')
        return parse_line(line)
    ```

## 4. Derived Features
Ensure your hardware firmware or middleware calculates these before inference:
1.  **total_acc**: `sqrt(ax^2 + ay^2 + az^2)`
2.  **gyro_magnitude**: `sqrt(gx^2 + gy^2 + gz^2)`
3.  **jerk**: `abs(total_acc_current - total_acc_previous)`

## 5. Thresholds (Pre-tuned)
Based on our ML Validation, use these starting thresholds for the ESP32:
*   **Crash**: `total_acc > 15.0` OR `gyro_magnitude > 10.0`
*   **Fall**: `gyro_magnitude > 4.0`
*   **Hard Brake**: `acc_x < -4.5`
