# ============================================================
# REFINED REAL-TIME CRASH DETECTION INFERENCE
# Usage: python inference.py
# Simulates a continuous sensor stream using the FINAL model.
# ============================================================

import os
import time
import numpy as np
import joblib

# -- Paths ----------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Loading the REFINED model and features
MODEL_PATH = os.path.join(BASE_DIR, "models", "crash_detection_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "models", "refined_feature_names.txt")

LABEL_MAP  = {0: "NORMAL", 1: "BRAKE", 2: "FALL", 3: "CRASH"}
SEVERITY   = {0: "OK",     1: "WARN",  2: "HIGH", 3: "CRITICAL"}
ALERT_ON   = {1, 2, 3}  # labels that trigger alerts

# -- Load model & feature list --------------------------------
print("Loading refined model ...")
pipe = joblib.load(MODEL_PATH)
with open(FEAT_PATH, encoding="utf-8") as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

print(f"Model loaded. Expecting {len(FEATURE_COLS)} features: {FEATURE_COLS}")
print("Starting real-time inference (Ctrl+C to stop) ...\n")

# -- Global state for jerk computation ------------------------
last_total_acc = 0.0

# -- Sensor simulation helpers --------------------------------
rng = np.random.default_rng()

def simulate_sensor_window(event_type="normal"):
    """
    Simulates one 128-sample IMU window and returns 
    the statistical features the refined model expects.
    """
    global last_total_acc
    
    if event_type == "normal":
        ax_w = rng.normal(0.0,  0.05, 128)
        ay_w = rng.normal(0.0,  0.05, 128)
        az_w = rng.normal(0.0,  0.05, 128)
        gx_w = rng.normal(0.0,  0.02, 128)
        gy_w = rng.normal(0.0,  0.02, 128)
        gz_w = rng.normal(0.0,  0.02, 128)

    elif event_type == "brake":
        # Deceleration along X
        ax_w = rng.uniform(-10, -5,   128) + rng.normal(0, 0.5, 128)
        ay_w = rng.uniform(-1.5, 1.5, 128) + rng.normal(0, 0.3, 128)
        az_w = rng.uniform(8.5, 10.5, 128) + rng.normal(0, 0.3, 128)
        gx_w = rng.normal(0, 0.2, 128)
        gy_w = rng.normal(0, 0.2, 128)
        gz_w = rng.normal(0, 0.2, 128)

    elif event_type == "fall":
        # Tumbling + variable gravity
        ax_w = rng.uniform(-4, 4,  128) + rng.normal(0, 0.5, 128)
        ay_w = rng.uniform(-4, 4,  128) + rng.normal(0, 0.5, 128)
        az_w = rng.uniform(-2, 3,  128) + rng.normal(0, 0.5, 128)
        gx_w = rng.uniform(3, 8,   128) * rng.choice([-1, 1], 128)
        gy_w = rng.uniform(3, 8,   128) * rng.choice([-1, 1], 128)
        gz_w = rng.uniform(2, 6,   128) * rng.choice([-1, 1], 128)

    elif event_type == "crash":
        # Extreme impact
        ax_w = rng.uniform(20, 45, 128) * rng.choice([-1, 1], 128)
        ay_w = rng.uniform(20, 40, 128) * rng.choice([-1, 1], 128)
        az_w = rng.uniform(20, 50, 128) * rng.choice([-1, 1], 128)
        gx_w = rng.uniform(10, 20, 128) * rng.choice([-1, 1], 128)
        gy_w = rng.uniform(10, 18, 128) * rng.choice([-1, 1], 128)
        gz_w = rng.uniform(10, 16, 128) * rng.choice([-1, 1], 128)
    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    # Means of components (as per pipeline Step 3)
    ax_m, ay_m, az_m = np.mean(ax_w), np.mean(ay_w), np.mean(az_w)
    gx_m, gy_m, gz_m = np.mean(gx_w), np.mean(gy_w), np.mean(gz_w)

    # Derived features
    total_acc      = np.sqrt(ax_m**2 + ay_m**2 + az_m**2)
    gyro_magnitude = np.sqrt(gx_m**2 + gy_m**2 + gz_m**2)
    jerk           = abs(total_acc - last_total_acc)
    
    # Update state
    last_total_acc = total_acc

    row_data = {
        "acc_x": ax_m, "acc_y": ay_m, "acc_z": az_m,
        "gyro_x": gx_m, "gyro_y": gy_m, "gyro_z": gz_m,
        "total_acc": total_acc,
        "gyro_magnitude": gyro_magnitude,
        "jerk": jerk
    }
    
    return np.array([row_data[c] for c in FEATURE_COLS]).reshape(1, -1)

# -- Real-time scenario ---------------------------------------
SCENARIO = [
    ("normal", 4),
    ("brake",  2),
    ("normal", 2),
    ("fall",   2),
    ("normal", 2),
    ("crash",  2),
    ("normal", 4),
]

print(f"{'Time':>10} {'#':>5} {'True Type':>10} {'Prediction':>12} {'Conf':>8} {'Status':>10}")
print("-" * 70)

sample_id = 0
try:
    for event_type, count in SCENARIO:
        for _ in range(count):
            sample_id += 1
            X_live = simulate_sensor_window(event_type)
            
            # Prediction
            pred  = pipe.predict(X_live)[0]
            proba = pipe.predict_proba(X_live)[0]
            conf  = proba[pred] * 100
            
            label  = LABEL_MAP[pred]
            status = SEVERITY[pred]
            alert  = " !!! ALERT !!!" if pred in ALERT_ON else ""
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] #{sample_id:04d} | {event_type.upper():>9} | {label:>12} | {conf:6.1f}% | {status:>10}{alert}")
            
            time.sleep(0.4)

except KeyboardInterrupt:
    print("\nInference stopped.")
print("\nSimulation complete.")
