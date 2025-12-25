import torch
import numpy as np
import pandas as pd
from src.client.model import Autoencoder
from src.utils.preprocessing import Preprocessor
import os

class FlarePipeline:
    def __init__(self, model_path="c:/proj/flare/src/client/autoencoder.pth", scaler_path="c:/proj/flare/src/client/scaler.pkl", input_dim=13):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.input_dim = input_dim
        self.model = None
        self.preprocessor = Preprocessor()
        self.feature_names = [
            'flow_duration', 'flow_bytes_s', 'flow_pkts_s', 
            'flow_iat_mean', 'flow_iat_std', 
            'fwd_pkt_len_max', 'total_fwd_pkts', 'total_bwd_pkts',
            'syn_flag_count', 'rst_flag_count', 'fin_flag_count',
            'active_mean', 'idle_mean'
        ]
        
    def load_artifacts(self):
        """Loads model and scaler."""
        # Load Model
        self.model = Autoencoder(self.input_dim)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        # Load Scaler
        try:
            self.preprocessor.load(self.scaler_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Run training first.")

    def preprocess(self, df: pd.DataFrame):
        """Preprocesses new traffic data."""
        # 1. Handle Feature Selection & Clipping
        # We process a copy to avoid SettingWithCopy warnings or mutating original
        df_processed = df[self.feature_names].copy()
        for col in self.feature_names:
            df_processed[col] = df_processed[col].clip(lower=0)
            
        # 2. Transform using Preprocessor
        # preprocessor.transform expects a DF that might contain 'label' to drop.
        # But here 'df_processed' definitely does NOT have 'label' because we filtered it out.
        # So we pass target_col=None or handle it such that Preprocessor doesn't crash.
        # Looking at Preprocessor.transform:
        #   X = df.drop(columns=[target_col]) -> This crashes if target_col not present.
        
        # Workaround: We bypass Preprocessor.transform for just scaling, OR we invoke scaler directly.
        # Better: Invoke scaler directly as we have already prepped X.
        X_scaled = self.preprocessor.scaler.transform(df_processed)
        
        return X_scaled

    def detect_anomaly(self, X_input, threshold=None, z_score_mode=False, mean_loss=None, std_loss=None):
        """
        Runs inference and anomaly detection.
        Args:
            X_input: Preprocessed numpy array
            threshold: Anomaly threshold (MSE).
            z_score_mode: If True, uses Z-score of loss > threshold (e.g. 3.0)
            mean_loss, std_loss: Stats of normal traffic (required for Z-score)
        """
        if self.model is None:
            self.load_artifacts()
            
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            # MSE per sample
            loss = torch.mean((X_tensor - reconstructed)**2, dim=1).numpy()
            
        alerts = None
        if z_score_mode and mean_loss is not None and std_loss is not None:
            # Adaptive Thresholding: Check statistical deviation
            z_scores = (loss - mean_loss) / (std_loss + 1e-9)
            alerts = z_scores > threshold # threshold usually 3.0 for Z-score
        elif threshold is not None:
             # Static Thresholding
            alerts = loss > threshold
            
        return loss, alerts
        
    def compute_threshold_stats(self, X_val):
        """Computes mean and std of loss for normal traffic (Adaptive Thresholding)."""
        loss, _ = self.detect_anomaly(X_val)
        return np.mean(loss), np.std(loss)
        
    def compute_threshold(self, X_val, percentile=95):
        """Computes static percentile threshold."""
        loss, _ = self.detect_anomaly(X_val)
        return np.percentile(loss, percentile)
