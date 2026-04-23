# 🛡️ PRANA One: Adaptive Guardian Engine

**PRANA (Adaptive Guardian Engine)** is a high-performance, AI-driven safety ecosystem designed for connected riders. It uses machine learning to detect crashes, falls, and abnormal riding behavior in real-time, triggering a rapid SOS response and providing a central command center for safety monitoring.

---

## 🚀 Key Features

### 📱 Intelligent Mobile App
- **Real-time Detection**: Rule-based engine optimized from a Random Forest model.
- **30s SOS Countdown**: Intelligent window for riders to cancel false alerts.
- **GPS Integration**: Precise location capture using browser/device Geolocation.
- **Onboarding**: Seamless emergency contact management.

### 🖥️ Guardian Command Dashboard
- **Live Incident Map**: Interactive Leaflet.js map showing real-time crash locations.
- **System Logs**: Live technical logs for monitoring device connectivity.
- **Alert Feed**: Categorized alerts (Crash, Fall, Hard Braking) with "Locate" and "Resolve" actions.
- **Contact Sync**: Instant access to rider emergency contacts during an event.

### 🧠 Machine Learning Backend
- **Model**: Random Forest Classifier trained on IMU (Accelerometer + Gyroscope) data.
- **Accuracy**: 99.5% on validation datasets.
- **Pipeline**: Automated data preparation, feature engineering, and model training scripts.

---

## 📂 Project Structure

```text
prana/
├── app/                # Mobile interface (HTML/JS/Tailwind)
├── backend/            # Python ML pipeline & Simulation scripts
├── dashboard/          # Central monitoring hub
├── data/               # Raw and processed datasets
├── models/             # Trained ML model artifacts (.joblib)
├── plots/              # Performance visualizations (CM, Importance)
└── Hardware_Integration_Guide.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.8+
- Modern Web Browser (Chrome/Edge recommended)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/prana.git
cd prana

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Running the Demo
1. Start the local development server:
   ```bash
   python -m http.server 8000
   ```
2. Open the **App**: `http://localhost:8000/app/index.html`
3. Open the **Dashboard**: `http://localhost:8000/dashboard/index.html`

---

## 🔌 Hardware Integration
This project is designed to be hardware-agnostic. For instructions on connecting physical IMU sensors (MPU6050) via ESP32, please refer to the [Hardware Integration Guide](Hardware_Integration_Guide.md).

## 📊 Model Performance
The detection logic is based on a Random Forest model with the following metrics:
- **Accuracy**: ~99%
- **F1-Score**: High precision for "Crash" classes to minimize false positives.
- **Features Used**: Total Acceleration, Gyro Magnitude, X-Axis Jerk.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License
This project is licensed under the MIT License.
