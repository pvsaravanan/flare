# FLARE: Federated Learning for Anomaly Detection (IoT Security)

**FLARE** is a privacy-preserving Intrusion Detection System (IDS) that uses **Federated Learning** to train Deep Autoencoders on edge devices without sharing raw data, and **Explainable AI (SHAP)** to interpret the alerts.

---

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.8+
- [Optional] CUDA-enabled GPU (PyTorch will use it if available)

### 2. Clone & Environment

```bash
# Clone the repository
git clone https://github.com/your-username/flare.git
cd flare

# Create Virtual Environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn flwr shap streamlit plotly matplotlib requests
```

### 4. Download Dataset

We use the **CICIDS2017** dataset. Run the helper script to download and extract it automatically:

```powershell
$env:PYTHONPATH='.'; python download_data.py
```

_This will create a `data/traffic.csv` file._

---

## 🏃‍♂️ How to Run

FLARE consists of 3 components that must run simultaneously. Open **3 separate terminal windows**.

### Terminal 1: The Server (Aggregator)

The server coordinates the Federated Learning rounds.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/server/server.py
```

_Wait until you see "Flower server running"._

### Terminal 2: The Client (Edge Device)

The client loads local data, trains the model, and performs anomaly detection.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/client/app.py
```

_The client will connect to the server, download the global model, train on local data, and upload updated weights._

### Terminal 3: The Dashboard (Control Center)

The dashboard visualizes real-time traffic, anomalies, and explanations.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python -m streamlit run dashboard/dashboard.py
```

_Open the URL shown (usually `http://localhost:8501`)._

---

## 🎮 Interactive Demo Features

In the Dashboard:

1.  **Simulate Normal Traffic**: Click the Green button. The chart should be low/stable (Green dots).
2.  **Simulate Web Attack**: Click the Red button. The chart will spike (Red crosses) and Alerts will trigger.
3.  **Explainability**: Scroll down to see exactly _why_ an attack was detected (e.g., "High Flow Duration" or "Excessive SYN Flags").

---

## 📂 Project Structure

```
flare/
├── src/
│   ├── client/
│   │   ├── model.py        # Deep Autoencoder Architecture (PyTorch)
│   │   ├── pipeline.py     # Inference & Thresholding Logic
│   │   ├── app.py          # Flower Client & Differential Privacy
│   │   └── explain.py      # SHAP Explainability Wrapper
│   ├── server/
│   │   └── server.py       # Flower Server Strategy
│   └── utils/
│       ├── data_loader.py  # CICIDS2017 Data Parser
│       └── preprocessing.py# MinMax Scaler & Data Loaders
├── dashboard/
│   └── dashboard.py        # Streamlit + Plotly UI
├── data/                   # Dataset directory
└── download_data.py        # Dataset downloader script
```
