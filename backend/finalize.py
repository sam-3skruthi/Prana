# ============================================================
# FINAL INTEGRATION ANALYSIS - CRASH DETECTION
# Extracts thresholds, rules, and stats for mobile developers
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Paths & Settings
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "final_crash_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "crash_detection_model.pkl")
LABEL_MAP   = {0: "Normal", 1: "Brake", 2: "Fall", 3: "Crash"}

# ─────────────────────────────────────────────
# STEP 1: Load Model & Data
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading model and dataset")
print("=" * 65)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run refined_pipeline.py first.")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Re-run the same feature engineering as refined_pipeline.py
if "gyro_magnitude" not in df.columns:
    df["gyro_magnitude"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
if "jerk" not in df.columns:
    df["jerk"] = df["total_acc"].diff().fillna(0).abs()

FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", 
                "total_acc", "gyro_magnitude", "jerk"]

print(f"  Dataset samples: {len(df)}")
print(f"  Features loaded: {FEATURE_COLS}")

# ─────────────────────────────────────────────
# STEP 2: Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Feature Importance Analysis")
print("=" * 65)

# For Random Forest in a Pipeline, the classifier is named 'clf'
clf = model.named_steps['clf']
importances = clf.feature_importances_
feat_importance = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances})
feat_importance = feat_importance.sort_values(by='importance', ascending=False)

print("  Top Features:")
print(feat_importance.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 3: Practical Class Stats & Threshold Discovery
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Class-wise Data Distribution (Practical Insights)")
print("=" * 65)

stats_cols = ["total_acc", "acc_x", "gyro_magnitude", "jerk"]
stats_out = {}

for lbl_id, lbl_name in LABEL_MAP.items():
    print(f"\n  --- CLASS: {lbl_name.upper()} ---")
    class_df = df[df['label'] == lbl_id]
    
    # Calculate stats
    s = class_df[stats_cols].describe().loc[['min', 'max', 'mean']]
    # Also get 95th/5th percentiles for cleaner thresholds
    p95 = class_df[stats_cols].quantile(0.95)
    p05 = class_df[stats_cols].quantile(0.05)
    
    summary = pd.concat([s, p05.to_frame(name='5%').T, p95.to_frame(name='95%').T])
    print(summary.round(3).to_string())
    
    stats_out[lbl_id] = summary

# ─────────────────────────────────────────────
# STEP 4: Derive Rule-Based Logic
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Deriving Decision Rules")
print("=" * 65)

# Heuristic derivation based on stats:
# Crash: total_acc is extremely high.
# Brake: acc_x is extremely low (negative).
# Fall: gyro_magnitude is high but acc is lower than crash.

rules_desc = """
1. CRASH: total_acc > 15.0 OR gyro_magnitude > 8.0
2. BRAKE: acc_x < -4.0 AND total_acc < 12.0
3. FALL:  gyro_magnitude > 3.0 AND total_acc < 15.0
4. NORMAL: Everything else
"""
print(rules_desc)

def rule_based_classifier(row):
    ta = row['total_acc']
    gm = row['gyro_magnitude']
    ax = row['acc_x']
    
    # Priority 1: Crash (High intensity overrides all)
    if ta > 15.0 or gm > 10.0:
        return 3 # Crash
    # Priority 2: Fall (Tumbling motion)
    if gm > 4.0:
        return 2 # Fall
    # Priority 3: Brake (Deceleration)
    if ax < -4.5:
        return 1 # Brake
    
    return 0 # Normal

# ─────────────────────────────────────────────
# STEP 5: Validate Rules vs Model
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Rule-Based Validation")
print("=" * 65)

# Split for validation
X = df[FEATURE_COLS]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predictions
y_model_pred = model.predict(X_test)
y_rule_pred  = X_test.apply(rule_based_classifier, axis=1)

model_acc = accuracy_score(y_test, y_model_pred)
rule_acc  = accuracy_score(y_test, y_rule_pred)

print(f"  ML Model Accuracy       : {model_acc*100:.2f}%")
print(f"  Rule-Based Accuracy     : {rule_acc*100:.2f}%")
print("\n  --- Rule-Based Classification Report ---")
print(classification_report(y_test, y_rule_pred, target_names=target_names if 'target_names' in locals() else list(LABEL_MAP.values())))

# ─────────────────────────────────────────────
# STEP 6: Final Integration Summary (Flutter/Dart Format)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: MOBILE INTEGRATION SUMMARY")
print("=" * 65)

summ = f"""
REQUIRED INPUTS:
- Accelerometer (X, Y, Z) in m/s^2
- Gyroscope (X, Y, Z) in rad/s

FEATURE FORMULAS:
- total_acc = sqrt(ax*ax + ay*ay + az*az)
- gyro_mag  = sqrt(gx*gx + gy*gy + gz*gz)

RECOMMENDED THRESHOLDS:
- CRASH_ACC_THRESHOLD: 15.0
- CRASH_GYRO_THRESHOLD: 10.0
- FALL_GYRO_THRESHOLD: 4.0
- BRAKE_ACC_X_THRESHOLD: -4.5

-----------------------------------------------------------
FLUTTER (DART) IMPLEMENTATION SNIPPET:
-----------------------------------------------------------
int detectActivity(double ax, double ay, double az, double gx, double gy, double gz) {{
  double totalAcc = sqrt(ax*ax + ay*ay + az*az);
  double gyroMag  = sqrt(gx*gx + gy*gy + gz*gz);

  // 1. Check for Crash (Highest Intensity)
  if (totalAcc > 15.0 || gyroMag > 10.0) {{
    return 3; // CRASH
  }}
  
  // 2. Check for Fall (High rotation, moderate acceleration)
  if (gyroMag > 4.0) {{
    return 2; // FALL
  }}
  
  // 3. Check for Hard Braking (Significant negative X acceleration)
  if (ax < -4.5) {{
    return 1; // BRAKE
  }}

  return 0; // NORMAL
}}
-----------------------------------------------------------
"""
print(summ)

# Save the statistics and summary to files
with open(os.path.join(BASE_DIR, "app", "mobile_integration_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summ)

print(f"Integration summary saved to: {os.path.join(BASE_DIR, 'app', 'mobile_integration_summary.txt')}")
