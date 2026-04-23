# ============================================================
# CRASH DETECTION - FULL ML PIPELINE
# Train -> Evaluate -> Save Model -> Real-Time Inference
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.metrics            import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.pipeline           import Pipeline

# ─── paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "final_crash_dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_NAMES = {0: "Normal", 1: "Brake", 2: "Fall", 3: "Crash"}

# ==============================================================
# STEP 1 : Load Dataset
# ==============================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Class distribution:")
for lbl, cnt in df["label"].value_counts().sort_index().items():
    print(f"    {lbl} - {LABEL_NAMES[lbl]:6s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

# ==============================================================
# STEP 2 : Feature Engineering & Preprocessing
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: Feature engineering & preprocessing")
print("=" * 60)

# Add extra derived features
df["acc_xy_ratio"]    = df["acc_x"] / (df["acc_y"].replace(0, 1e-6))
df["gyro_magnitude"]  = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
df["acc_std_total"]   = np.sqrt(
    df["acc_x_std"]**2 + df["acc_y_std"]**2 + df["acc_z_std"]**2
)
df["gyro_std_total"]  = np.sqrt(
    df["gyro_x_std"]**2 + df["gyro_y_std"]**2 + df["gyro_z_std"]**2
)

# Clip extreme outliers to 3-sigma per feature (keeps synthetic crash data realistic)
FEATURE_COLS = [c for c in df.columns if c != "label"]
for col in FEATURE_COLS:
    mu, sigma = df[col].mean(), df[col].std()
    df[col] = df[col].clip(mu - 4*sigma, mu + 4*sigma)

X = df[FEATURE_COLS].values
y = df["label"].values

print(f"  Features used : {len(FEATURE_COLS)}")
print(f"  Feature names : {FEATURE_COLS}")
print(f"  X shape       : {X.shape}")
print(f"  y shape       : {y.shape}")

# ==============================================================
# STEP 3 : Train / Test Split
# ==============================================================
print("\n" + "=" * 60)
print("STEP 3: Train/test split (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

# ==============================================================
# STEP 4 : Train Multiple Models
# ==============================================================
print("\n" + "=" * 60)
print("STEP 4: Training models")
print("=" * 60)

models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ]),
}

results    = {}
cv_scores  = {}
skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models.items():
    print(f"\n  Training: {name} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="weighted")
    cv    = cross_val_score(pipe, X_train, y_train, cv=skf,
                            scoring="accuracy", n_jobs=-1).mean()

    results[name]   = {"pipe": pipe, "y_pred": y_pred, "acc": acc, "f1": f1}
    cv_scores[name] = cv
    print(f"    Test Accuracy  : {acc*100:.2f}%")
    print(f"    Weighted F1    : {f1:.4f}")
    print(f"    CV Accuracy    : {cv*100:.2f}%")

# ==============================================================
# STEP 5 : Evaluate Best Model
# ==============================================================
print("\n" + "=" * 60)
print("STEP 5: Model comparison & full evaluation of best model")
print("=" * 60)

# Pick model with highest test accuracy
best_name = max(results, key=lambda k: results[k]["acc"])
best      = results[best_name]
best_pipe = best["pipe"]
y_pred    = best["y_pred"]

print(f"\n  Best Model: {best_name}")
print(f"  Test Accuracy: {best['acc']*100:.2f}%")
print(f"  Weighted F1  : {best['f1']:.4f}")

print("\n  --- Model Comparison Table ---")
print(f"  {'Model':<22} {'Test Acc':>10} {'F1':>8} {'CV Acc':>10}")
print("  " + "-" * 54)
for name in results:
    r = results[name]
    print(f"  {name:<22} {r['acc']*100:>9.2f}% {r['f1']:>8.4f} {cv_scores[name]*100:>9.2f}%")

print("\n  --- Classification Report ---")
label_order = sorted(LABEL_NAMES.keys())
target_names = [LABEL_NAMES[l] for l in label_order]
print(classification_report(y_test, y_pred, labels=label_order,
                             target_names=target_names))

print("  --- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=label_order)
header = f"  {'':>10}" + "".join(f"{LABEL_NAMES[l]:>10}" for l in label_order)
print(header)
for i, row in enumerate(cm):
    row_str = f"  {LABEL_NAMES[label_order[i]]:>10}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# Feature importance (RandomForest/GB only)
if hasattr(best_pipe.named_steps["clf"], "feature_importances_"):
    fi = best_pipe.named_steps["clf"].feature_importances_
    fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi})
    fi_df = fi_df.sort_values("importance", ascending=False)
    print("\n  --- Top 10 Feature Importances ---")
    print(f"  {'Feature':<25} {'Importance':>12}")
    print("  " + "-" * 39)
    for _, row in fi_df.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:>12.4f}")

# ==============================================================
# STEP 6 : Save Model + Scaler + Metadata
# ==============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving model artifacts")
print("=" * 60)

model_path    = os.path.join(MODEL_DIR, "best_crash_model.pkl")
meta_path     = os.path.join(MODEL_DIR, "model_metadata.txt")
features_path = os.path.join(MODEL_DIR, "feature_names.txt")

# Save full pipeline (includes scaler)
joblib.dump(best_pipe, model_path)
print(f"  Pipeline saved  : {model_path}")

# Save feature names
with open(features_path, "w", encoding="utf-8") as f:
    f.write("\n".join(FEATURE_COLS))
print(f"  Feature names   : {features_path}")

# Save metadata
with open(meta_path, "w", encoding="utf-8") as f:
    f.write(f"Best Model      : {best_name}\n")
    f.write(f"Test Accuracy   : {best['acc']*100:.2f}%\n")
    f.write(f"Weighted F1     : {best['f1']:.4f}\n")
    f.write(f"CV Accuracy     : {cv_scores[best_name]*100:.2f}%\n")
    f.write(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")
    f.write(f"Label Map       : {LABEL_NAMES}\n")
print(f"  Metadata saved  : {meta_path}")

# ==============================================================
# STEP 7 : Generate Real-Time Inference Script
# ==============================================================
print("\n" + "=" * 60)
print("STEP 7: Generating real-time inference script ...")
print("=" * 60)

INFERENCE_SCRIPT = '''# ============================================================
# REAL-TIME CRASH DETECTION INFERENCE
# Usage: python inference.py
# Simulates a continuous sensor stream and classifies each
# window of readings into Normal / Brake / Fall / Crash.
# ============================================================

import os
import time
import numpy as np
import joblib

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_crash_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "models", "feature_names.txt")

LABEL_MAP  = {0: "NORMAL", 1: "BRAKE", 2: "FALL", 3: "CRASH"}
SEVERITY   = {0: "OK",     1: "WARN",  2: "HIGH", 3: "CRITICAL"}
ALERT_ON   = {1, 2, 3}  # labels that trigger alerts

# ── Load model & feature list ───────────────────────────────
print("Loading model ...")
pipe     = joblib.load(MODEL_PATH)
with open(FEAT_PATH) as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

print(f"Model loaded. Expecting {len(FEATURE_COLS)} features.")
print("Starting real-time inference (Ctrl+C to stop) ...\\n")

# ── Sensor simulation helpers ───────────────────────────────
rng = np.random.default_rng()

def simulate_sensor_window(event_type="normal"):
    """
    Simulates one 128-sample IMU window and returns
    the same statistical features the model was trained on.
    Replace this function with actual sensor reads in production.
    """
    if event_type == "normal":
        ax_w = rng.normal(0.0,  0.05, 128)
        ay_w = rng.normal(0.0,  0.05, 128)
        az_w = rng.normal(0.0,  0.05, 128)
        gx_w = rng.normal(0.0,  0.02, 128)
        gy_w = rng.normal(0.0,  0.02, 128)
        gz_w = rng.normal(0.0,  0.02, 128)

    elif event_type == "brake":
        ax_w = rng.uniform(-10, -5,  128) + rng.normal(0, 0.5, 128)
        ay_w = rng.uniform(-1.5, 1.5, 128) + rng.normal(0, 0.3, 128)
        az_w = rng.uniform(8.5, 10.5, 128) + rng.normal(0, 0.3, 128)
        gx_w = rng.uniform(-1, 1,    128) + rng.normal(0, 0.2, 128)
        gy_w = rng.uniform(-0.5, 0.5, 128)
        gz_w = rng.uniform(-0.5, 0.5, 128)

    elif event_type == "fall":
        ax_w = rng.uniform(-4, 4,   128) + rng.normal(0, 0.5, 128)
        ay_w = rng.uniform(-4, 4,   128) + rng.normal(0, 0.5, 128)
        az_w = rng.uniform(-2, 3,   128) + rng.normal(0, 0.5, 128)
        gx_w = rng.uniform(3, 8,    128) * rng.choice([-1,1], 128)
        gy_w = rng.uniform(3, 8,    128) * rng.choice([-1,1], 128)
        gz_w = rng.uniform(2, 6,    128) * rng.choice([-1,1], 128)

    elif event_type == "crash":
        ax_w = rng.uniform(20, 45,  128) * rng.choice([-1,1], 128)
        ay_w = rng.uniform(20, 40,  128) * rng.choice([-1,1], 128)
        az_w = rng.uniform(20, 50,  128) * rng.choice([-1,1], 128)
        gx_w = rng.uniform(10, 20,  128) * rng.choice([-1,1], 128)
        gy_w = rng.uniform(10, 18,  128) * rng.choice([-1,1], 128)
        gz_w = rng.uniform(10, 16,  128) * rng.choice([-1,1], 128)

    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    def feats(w): return np.mean(w), np.std(w)

    ax_m, ax_s = feats(ax_w)
    ay_m, ay_s = feats(ay_w)
    az_m, az_s = feats(az_w)
    gx_m, gx_s = feats(gx_w)
    gy_m, gy_s = feats(gy_w)
    gz_m, gz_s = feats(gz_w)

    total_acc     = np.sqrt(ax_m**2 + ay_m**2 + az_m**2)
    acc_xy_ratio  = ax_m / (ay_m if abs(ay_m) > 1e-6 else 1e-6)
    gyro_mag      = np.sqrt(gx_m**2 + gy_m**2 + gz_m**2)
    acc_std_total = np.sqrt(ax_s**2 + ay_s**2 + az_s**2)
    gyro_std_tot  = np.sqrt(gx_s**2 + gy_s**2 + gz_s**2)

    # Must match FEATURE_COLS order from training
    row = {
        "acc_x": ax_m, "acc_y": ay_m, "acc_z": az_m,
        "gyro_x": gx_m, "gyro_y": gy_m, "gyro_z": gz_m,
        "acc_x_std": ax_s, "acc_y_std": ay_s, "acc_z_std": az_s,
        "gyro_x_std": gx_s, "gyro_y_std": gy_s, "gyro_z_std": gz_s,
        "total_acc": total_acc,
        "acc_xy_ratio": acc_xy_ratio,
        "gyro_magnitude": gyro_mag,
        "acc_std_total": acc_std_total,
        "gyro_std_total": gyro_std_tot,
    }
    return np.array([row[c] for c in FEATURE_COLS]).reshape(1, -1)

# ── Scenario sequence (simulates a drive) ──────────────────
SCENARIO = [
    ("normal", 6),
    ("brake",  2),
    ("normal", 4),
    ("fall",   2),
    ("normal", 3),
    ("crash",  2),
    ("normal", 5),
]

# ── Inference loop ──────────────────────────────────────────
sample_id = 0
try:
    for event_type, repeats in SCENARIO:
        for _ in range(repeats):
            sample_id += 1
            X_live  = simulate_sensor_window(event_type)
            pred    = pipe.predict(X_live)[0]
            proba   = pipe.predict_proba(X_live)[0]
            conf    = proba[pred] * 100
            status  = SEVERITY[pred]
            label   = LABEL_MAP[pred]

            timestamp = time.strftime("%H:%M:%S")
            alert = " <<< ALERT!" if pred in ALERT_ON else ""

            print(
                f"[{timestamp}] Sample #{sample_id:04d} | "
                f"True: {event_type.upper():<7} | "
                f"Pred: {label:<9} | "
                f"Conf: {conf:5.1f}% | "
                f"Status: {status}{alert}"
            )
            time.sleep(0.3)   # 300ms per window (simulates 128-sample @ ~400Hz)

except KeyboardInterrupt:
    print("\\nInference stopped by user.")

print("\\nDone.")
'''

inference_path = os.path.join(BASE_DIR, "inference.py")
with open(inference_path, "w", encoding="utf-8") as f:
    f.write(INFERENCE_SCRIPT)
print(f"  Inference script : {inference_path}")

# ==============================================================
# DONE
# ==============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Dataset      : {DATA_PATH}")
print(f"  Best model   : {best_name}  ({best['acc']*100:.2f}% accuracy)")
print(f"  Saved model  : {model_path}")
print(f"  Inference    : {inference_path}")
print("\n  Run inference with:")
print("    python inference.py")
print("=" * 60)
