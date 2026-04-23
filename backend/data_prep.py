# ============================================================
# UCI HAR -> Crash Detection Dataset Generator
# Reads UCI_HAR.zip directly from the local folder
# ============================================================

import zipfile
import os
import numpy as np
import pandas as pd

# ---------------------------------------------
# STEP 1: Locate & Extract ZIP
# ---------------------------------------------
print("=" * 60)
print("STEP 1: Locating and extracting UCI_HAR.zip")
print("=" * 60)

# Path to the ZIP file (in data/raw/)
script_dir   = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
zip_filename = os.path.join(script_dir, "..", "data", "raw", "UCI_HAR.zip")
extract_dir  = os.path.join(script_dir, "..", "data", "raw", "UCI_HAR_extracted")

if not os.path.isfile(zip_filename):
    raise FileNotFoundError(f"ZIP not found: {zip_filename}")
print(f"Found: {zip_filename}")

print(f"Extracting to: {extract_dir} ...")
with zipfile.ZipFile(zip_filename, 'r') as zf:
    zf.extractall(extract_dir)
print("Extraction complete.\n")

# ---------------------------------------------
# STEP 2: Read Inertial Signals from Train Set
# ---------------------------------------------
print("=" * 60)
print("STEP 2: Loading inertial signals from train set")
print("=" * 60)

INERTIAL_BASE = os.path.join(extract_dir, "UCI-HAR Dataset", "train", "Inertial Signals")
LABEL_PATH    = os.path.join(extract_dir, "UCI-HAR Dataset", "train", "y_train.txt")

signal_files = {
    "acc_x":  "body_acc_x_train.txt",
    "acc_y":  "body_acc_y_train.txt",
    "acc_z":  "body_acc_z_train.txt",
    "gyro_x": "body_gyro_x_train.txt",
    "gyro_y": "body_gyro_y_train.txt",
    "gyro_z": "body_gyro_z_train.txt",
}

raw_signals = {}
for col_name, fname in signal_files.items():
    fpath = os.path.join(INERTIAL_BASE, fname)
    # Each row has 128 time-step readings separated by spaces
    data = pd.read_csv(fpath, sep=r'\s+', header=None)
    raw_signals[col_name] = data
    print(f"  Loaded {fname:35s} -> shape {data.shape}")

# Load activity labels (1-6)
y_train_raw = pd.read_csv(LABEL_PATH, header=None, names=["activity"])
print(f"\n  Loaded y_train.txt -> shape {y_train_raw.shape}")
print(f"  Unique activities: {sorted(y_train_raw['activity'].unique())}")

# ---------------------------------------------
# STEP 3: Convert Time-Series Windows -> Features
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Extracting statistical features from windows")
print("=" * 60)

def extract_features(arr):
    """Return mean and std across the 128-step window for each sample."""
    return pd.DataFrame({
        "mean": arr.mean(axis=1).values,
        "std":  arr.std(axis=1).values,
    })

# Use only the MEAN as the representative value per window
# (keeps the dataset simple; std is computed and stored separately)
feature_records = {}
for col_name, data in raw_signals.items():
    feature_records[col_name]           = data.mean(axis=1).values  # mean over window
    feature_records[f"{col_name}_std"]  = data.std(axis=1).values   # window std

har_df = pd.DataFrame(feature_records)
print(f"  Feature matrix shape: {har_df.shape}")

# ---------------------------------------------
# STEP 4: Label Mapping -> All HAR = Normal (0)
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Mapping HAR activities -> Normal (label = 0)")
print("=" * 60)

activity_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
har_df["label"] = y_train_raw["activity"].map(activity_map).values
print(f"  Total normal samples available: {len(har_df)}")

# Use at least 2000 normal samples
NORMAL_COUNT = max(2000, len(har_df))
normal_df = har_df.sample(n=min(NORMAL_COUNT, len(har_df)), random_state=42).reset_index(drop=True)
print(f"  Using {len(normal_df)} normal samples")

# ---------------------------------------------
# STEP 5: Generate Synthetic Data
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Generating synthetic BRAKE / FALL / CRASH samples")
print("=" * 60)

rng = np.random.default_rng(seed=2024)

def make_noise(size, scale=0.2):
    return rng.normal(0, scale, size)

# -- 5a. BRAKE (label = 1) ------------------------------------
N_BRAKE = 500
brake = pd.DataFrame()

# Strong negative deceleration along x-axis (-5 to -10)
brake["acc_x"]     = rng.uniform(-10.0, -5.0, N_BRAKE) + make_noise(N_BRAKE, 0.5)
# Slight lateral sway during braking
brake["acc_y"]     = rng.uniform(-1.5,   1.5, N_BRAKE) + make_noise(N_BRAKE, 0.3)
# Vertical stays near gravity (9.8) with small pitch
brake["acc_z"]     = rng.uniform(8.5,   10.5, N_BRAKE) + make_noise(N_BRAKE, 0.3)
# Small gyroscope variation (pitch forward slightly)
brake["gyro_x"]    = rng.uniform(-1.0,   1.0, N_BRAKE) + make_noise(N_BRAKE, 0.2)
brake["gyro_y"]    = rng.uniform(-0.5,   0.5, N_BRAKE) + make_noise(N_BRAKE, 0.15)
brake["gyro_z"]    = rng.uniform(-0.5,   0.5, N_BRAKE) + make_noise(N_BRAKE, 0.15)
# Std features: moderate noise
for ch in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
    brake[f"{ch}_std"] = rng.uniform(0.3, 1.5, N_BRAKE)
brake["label"]     = 1
print(f"  BRAKE  samples generated: {len(brake)}")

# -- 5b. FALL (label = 2) -------------------------------------
N_FALL = 500
fall = pd.DataFrame()

# Freefall: acc_z deviates significantly from 9.8
fall["acc_x"]      = rng.uniform(-4.0,  4.0,  N_FALL) + make_noise(N_FALL, 0.5)
fall["acc_y"]      = rng.uniform(-4.0,  4.0,  N_FALL) + make_noise(N_FALL, 0.5)
# acc_z: near 0 during freefall, or high on impact
fall_phase = rng.choice(["freefall", "impact"], N_FALL, p=[0.6, 0.4])
fall["acc_z"]      = np.where(
    fall_phase == "freefall",
    rng.uniform(-2.0,  3.0, N_FALL),    # near-zero gravity
    rng.uniform(15.0, 25.0, N_FALL)     # impact spike
) + make_noise(N_FALL, 0.5)
# Gyro spikes 3-8 rad/s during tumble
fall["gyro_x"]     = rng.uniform(3.0, 8.0, N_FALL) * rng.choice([-1, 1], N_FALL) + make_noise(N_FALL, 0.4)
fall["gyro_y"]     = rng.uniform(3.0, 8.0, N_FALL) * rng.choice([-1, 1], N_FALL) + make_noise(N_FALL, 0.4)
fall["gyro_z"]     = rng.uniform(2.0, 6.0, N_FALL) * rng.choice([-1, 1], N_FALL) + make_noise(N_FALL, 0.3)
# High std - turbulent signal
for ch in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
    fall[f"{ch}_std"] = rng.uniform(1.5, 4.0, N_FALL)
fall["label"]      = 2
print(f"  FALL   samples generated: {len(fall)}")

# -- 5c. CRASH (label = 3) ------------------------------------
N_CRASH = 500
crash = pd.DataFrame()

# Phase split: high impact then near-zero stillness
crash_phase = rng.choice(["impact", "post"], N_CRASH, p=[0.65, 0.35])

# Impact: acc > 20 in all axes; post-crash: near zero
crash["acc_x"]  = np.where(
    crash_phase == "impact",
    rng.uniform(20.0, 45.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform(-0.5,  0.5, N_CRASH)
) + make_noise(N_CRASH, 0.8)

crash["acc_y"]  = np.where(
    crash_phase == "impact",
    rng.uniform(20.0, 40.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform(-0.5,  0.5, N_CRASH)
) + make_noise(N_CRASH, 0.8)

crash["acc_z"]  = np.where(
    crash_phase == "impact",
    rng.uniform(20.0, 50.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform( 8.5, 10.5, N_CRASH)   # gravity restored when still
) + make_noise(N_CRASH, 0.8)

# Gyro: > 10 rad/s on impact, near 0 post-crash
crash["gyro_x"] = np.where(
    crash_phase == "impact",
    rng.uniform(10.0, 20.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform(-0.3,  0.3, N_CRASH)
) + make_noise(N_CRASH, 0.4)

crash["gyro_y"] = np.where(
    crash_phase == "impact",
    rng.uniform(10.0, 18.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform(-0.3,  0.3, N_CRASH)
) + make_noise(N_CRASH, 0.4)

crash["gyro_z"] = np.where(
    crash_phase == "impact",
    rng.uniform(10.0, 16.0, N_CRASH) * rng.choice([-1, 1], N_CRASH),
    rng.uniform(-0.3,  0.3, N_CRASH)
) + make_noise(N_CRASH, 0.4)

# Std: very high at impact, very low post-crash
crash_std_scale = np.where(crash_phase == "impact",
                           rng.uniform(5.0, 12.0, N_CRASH),
                           rng.uniform(0.0,  0.3, N_CRASH))
for ch in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
    crash[f"{ch}_std"] = crash_std_scale + rng.normal(0, 0.2, N_CRASH)

crash["label"] = 3
print(f"  CRASH  samples generated: {len(crash)}")

# ---------------------------------------------
# STEP 6: Combine Real + Synthetic
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Combining all data")
print("=" * 60)

# Align column order
COLUMNS = ["acc_x", "acc_y", "acc_z",
           "gyro_x", "gyro_y", "gyro_z",
           "acc_x_std", "acc_y_std", "acc_z_std",
           "gyro_x_std", "gyro_y_std", "gyro_z_std",
           "label"]

# Ensure std cols exist on normal_df
for ch in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
    if f"{ch}_std" not in normal_df.columns:
        normal_df[f"{ch}_std"] = 0.0

normal_df = normal_df[COLUMNS]
brake     = brake[COLUMNS]
fall      = fall[COLUMNS]
crash     = crash[COLUMNS]

combined = pd.concat([normal_df, brake, fall, crash], ignore_index=True)
print(f"  Combined shape before feature engineering: {combined.shape}")

label_counts = combined["label"].value_counts().sort_index()
label_names  = {0: "Normal", 1: "Brake", 2: "Fall", 3: "Crash"}
for lbl, cnt in label_counts.items():
    print(f"    Label {lbl} ({label_names[lbl]:6s}): {cnt:5d} samples")

# ---------------------------------------------
# STEP 7: Engineered Feature - total_acc
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Adding engineered feature 'total_acc'")
print("=" * 60)

combined["total_acc"] = np.sqrt(
    combined["acc_x"]**2 +
    combined["acc_y"]**2 +
    combined["acc_z"]**2
)
print(f"  total_acc range: [{combined['total_acc'].min():.3f}, {combined['total_acc'].max():.3f}]")

# ---------------------------------------------
# STEP 8: Shuffle Dataset
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Shuffling dataset")
print("=" * 60)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Dataset shuffled. Final shape: {combined.shape}")

# ---------------------------------------------
# STEP 9: Save CSV
# ---------------------------------------------
OUTPUT_FILE = os.path.join(script_dir, "..", "data", "final_crash_dataset.csv")
combined.to_csv(OUTPUT_FILE, index=False)
print("\n" + "=" * 60)
print(f"STEP 9: Saved -> {OUTPUT_FILE}")
print("=" * 60)

# ---------------------------------------------
# STEP 10: Summary Report
# ---------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\n  Dataset shape : {combined.shape}")
print(f"  Columns       : {list(combined.columns)}\n")

print("  Class distribution:")
for lbl, cnt in combined["label"].value_counts().sort_index().items():
    pct = cnt / len(combined) * 100
    print(f"    Label {lbl} ({label_names[lbl]:6s}): {cnt:5d}  ({pct:.1f}%)")

print("\n  First 5 rows:")
print(combined.head().to_string(index=False))

print("\n  Descriptive statistics (main channels):")
print(combined[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","total_acc"]].describe().round(3).to_string())

print(f"\n  File saved at: {os.path.abspath(OUTPUT_FILE)}")
