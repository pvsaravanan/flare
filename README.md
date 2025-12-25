# FLARE: Federated Learning for Anomaly Detection

FLARE is a privacy-preserving Intrusion Detection System (IDS) for IoT networks using Federated Learning and Explainable AI.

## Architecture

- **Server**: Orchestrates training rounds (Flower).
- **Client**: Trains local Autoencoder on real traffic data (CICIDS2017) and detects anomalies.
- **Dashboard**: Visualizes traffic, alerts, and SHAP explanations.

## Setup

1. Install dependencies (if not already installed):
   ```bash
   pip install flwr torch torchvision numpy pandas matplotlib scikit-learn shap streamlit
   ```

## Running the Application

You will need **3 separate terminal windows**.

### 1. Start the Federated Server

The server coordinates the learning process.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/server/server.py
```

### 2. Start the Federated Client

The client loads local data, trains the model, and performs inference.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/client/app.py
```

_Note: You can run multiple clients in different terminals to simulate a distributed network._

### 3. Launch the Dashboard

The dashboard provides a real-time view of the system's status.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python -m streamlit run dashboard/dashboard.py
```

## Data

The system automatically uses the **CICIDS2017** dataset located in `data/traffic.csv`.
