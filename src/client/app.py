import flwr as fl
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from src.utils.data_loader import TrafficLoader
from src.utils.preprocessing import Preprocessor, get_dataloaders
from src.client.model import Autoencoder, get_loss_criterion
from src.client.explain import Explainer
from src.client.pipeline import FlarePipeline

# 1. Load Data (Simulate local client data)
# Each client generates its own data
simulator = TrafficLoader(n_samples=500)
df = simulator.generate_data()
preprocessor = Preprocessor()
# Train on NORMAL only
normal_df = df[df['label'] == 0]
X_train, y_train = preprocessor.fit_transform(normal_df)
# Test on all (to check reconstruction error on anomalies)
X_test, y_test = preprocessor.transform(df)

train_loader, _ = get_dataloaders(X_train, y_train, batch_size=32)
# test_loader not strictly used in current fit logic but useful for eval

# 2. Define Model
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)
criterion = get_loss_criterion()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Define Flower Client
class FlareClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Differential Privacy: Add Gaussian Noise
        params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        noise_multiplier = 0.01 # Adjust for privacy budget
        noisy_params = [p + np.random.normal(0, noise_multiplier, p.shape) for p in params]
        return noisy_params

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for epoch in range(1): # 1 local epoch per round
            for data, _ in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        loss = 0.0
        with torch.no_grad():
            for data, _ in train_loader: # Evaluate on "Normal" data usually to check convergence
                 outputs = model(data)
                 loss += criterion(outputs, data).item() * data.size(0)
        loss /= len(train_loader.dataset)
        return loss, len(train_loader.dataset), {"mse": loss}

# 4. Integrate Explainability using Pipeline Logic for Demo
def explain_anomaly_demo():
    print("Running Explanation Demo using FlarePipeline...")
    try:
        # Init Pipeline (Assumes model/scaler saved by train.py or previous rounds)
        # For this demo, we can just point to the artifacts
        pipeline = FlarePipeline()
        try:
            pipeline.load_artifacts()
        except Exception as e:
            print(f"Pipeline Load Error: {e}")
            return

        # Generate new data from real set (subsample)
        sim = TrafficLoader(n_samples=50)
        df = sim.generate_data()
        
        X = pipeline.preprocess(df)
        
        # Detect
        loss, _ = pipeline.detect_anomaly(X)
        print(f"DEBUG: Loss shape: {loss.shape}")
        
        # Pick anomaly
        idx = np.argmax(loss)
        print(f"Max Anomaly Score: {loss[idx]:.4f}")
        
        # Explain
        # Re-use explainer logic but with pipeline's loaded model
        # Need background data. Pipeline doesn't store data, so we need to provide it or store a representative set.
        # For demo, we use a slice of current X as background (assuming some normal traffic)
        bg = torch.tensor(X[:10], dtype=torch.float32)
        explainer = Explainer(pipeline.model, bg)
        
        shap_vals = explainer.explain(X[idx:idx+1])
        text = explainer.generate_text_explanation(shap_vals, pipeline.feature_names)
        print("Explanation:", text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")

def start_client():
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlareClient())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "explain":
        explain_anomaly_demo()
    else:
        start_client()
