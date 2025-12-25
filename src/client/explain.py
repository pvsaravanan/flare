import shap
import torch
import numpy as np
import pandas as pd

class Explainer:
    def __init__(self, model, background_data):
        """
        Args:
            model: PyTorch model
            background_data: A representative sample of the training data (numpy or tensor)
                             used by SHAP to integrate over.
        """
        self.model = model
        self.model.eval()
        
        # Convert background to tensor if needed
        if isinstance(background_data, np.ndarray):
            background_data = torch.tensor(background_data, dtype=torch.float32)
            
        # DeepExplainer is often preferred for DL, but KernelExplainer is more robust for generic functions.
        # Ideally use DeepExplainer for PyTorch, but sometimes version specific. 
        # For simplicity and robustness here, using KernelExplainer wrapped around a predict function.
        self.background_data = background_data
        
        # Wrapper for KernelExplainer to accept numpy arrays
        def predict_fn(x_np):
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            with torch.no_grad():
                reconstructed = self.model(x_tensor)
                # For anomaly detection, we might want to explain the Reconstruction ERROR, 
                # effectively model(x) - x. 
                # But typically we explain the output (reconstruction).
                # To explain anomaly score (MSE), we need a custom function.
                loss = torch.mean((x_tensor - reconstructed)**2, dim=1)
            return loss.numpy()

        # Using KernelExplainer to explain the 'Anomaly Score' (MSE Loss) directly
        # This tells us: "Feature X increased the anomaly score by Y"
        self.explainer = shap.KernelExplainer(predict_fn, self.background_data.numpy())

    def explain(self, sample):
        """
        Generate SHAP values for a single sample (or batch).
        Args:
            sample: numpy array (1, n_features)
        Returns:
            shap_values: importance of each feature for the anomaly score
        """
        shap_values = self.explainer.shap_values(sample)
        return shap_values

    def generate_text_explanation(self, shap_values, feature_names, threshold=0.1):
        """
        Generates a human-readable explanation from SHAP values.
        Alerts on features that pushed the score HIGHER (positive SHAP).
        """
        # shap_values might be a list or array depending on Explainer
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Flatten if needed
        if len(vals.shape) > 1:
            vals = vals[0]
            
        contributors = []
        for name, val in zip(feature_names, vals):
            if val > 0: # Only care about what INCREASED the anomaly score
                contributors.append((name, val))
        
        pass  # Sort by importance
        contributors.sort(key=lambda x: x[1], reverse=True)
        
        if not contributors:
            return "No specific features contributed significantly to the anomaly score."
            
        top_contributors = contributors[:3] # Top 3
        reasons = [f"'{x[0]}' (impact: {x[1]:.4f})" for x in top_contributors]
        
        return f"Anomaly detected! Primary contributors: {', '.join(reasons)}."
