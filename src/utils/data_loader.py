import numpy as np
import pandas as pd
import os

class TrafficLoader:
    def __init__(self, n_samples=None, anomaly_ratio=None):
        """
        n_samples: If set, subsamples the dataset.
        anomaly_ratio: Ignored for real data (determined by dataset), 
                       or used to downsample majority class if advanced logic needed. 
                       Kept for API compatibility but generally not used to force ratio.
        """
        self.n_samples = n_samples
        self.anomaly_ratio = anomaly_ratio # Stored for filtering logic
        self.feature_map = {
            'Flow Duration': 'flow_duration', 
            'Flow Bytes/s': 'flow_bytes_s', 
            'Flow Packets/s': 'flow_pkts_s', 
            'Flow IAT Mean': 'flow_iat_mean', 
            'Flow IAT Std': 'flow_iat_std', 
            'Fwd Packet Length Max': 'fwd_pkt_len_max', 
            'Total Fwd Packets': 'total_fwd_pkts', 
            'Total Backward Packets': 'total_bwd_pkts',
            'SYN Flag Count': 'syn_flag_count', 
            'RST Flag Count': 'rst_flag_count', 
            'FIN Flag Count': 'fin_flag_count',
            'Active Mean': 'active_mean', 
            'Idle Mean': 'idle_mean',
            'Label': 'label'
        }
        self.final_features = list(self.feature_map.values())
        self.final_features.remove('label')

    def load_data(self, path="c:/proj/flare/data/traffic.csv"):
        """Loads and processes the real CSV data."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}. Please run download script.")
            
        # Read CSV
        # CICIDS can have infinity or nan, handle them
        # Use latin-1 or handle errors for robustness
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
             print("Warning: UTF-8 decode failed, trying latin-1")
             df = pd.read_csv(path, encoding='latin-1')
        
        # Strip whitespace from columns
        df.columns = df.columns.str.strip()
        
        # Select and Rename Columns
        # Filter for columns we want (robustly)
        available_cols = [c for c in self.feature_map.keys() if c in df.columns]
        df = df[available_cols]
        df = df.rename(columns=self.feature_map)
        
        # Clean Data
        # 1. Replace Infinity with Max Float or NaN then Drop
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # 2. Encode Label
        # BENIGN -> 0, Everything else -> 1
        if 'label' in df.columns:
            df['label'] = df['label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # 3. Handle Strict Label Filtering (For Demo Buttons)
        # If anomaly_ratio is 0.0 -> Force only Benign (Label 0)
        # If anomaly_ratio is 1.0 -> Force only Attack (Label 1)
        # If anomaly_ratio is None or other -> Normal sampling
        
        if self.anomaly_ratio is not None:
            if self.anomaly_ratio == 0.0:
                print("Forcing Purely Normal Traffic (Demo Mode)")
                df = df[df['label'] == 0]
            elif self.anomaly_ratio == 1.0:
                 print("Forcing Purely Attack Traffic (Demo Mode)")
                 df = df[df['label'] == 1]
        
        # 4. Subsample if requested
        if self.n_samples and self.n_samples < len(df):
            # Try to maintain some anomalies if possible, or just random sample
            df = df.sample(n=self.n_samples, random_state=42)
            
        print(f"Loaded {len(df)} samples from {path}")
        return df
        
    def generate_data(self):
        """API wrapper for compatibility. Calls load_data."""
        return self.load_data()
        
    @property
    def feature_names(self):
        return self.final_features

if __name__ == "__main__":
    loader = TrafficLoader(n_samples=1000)
    df = loader.load_data()
    print(df.head())
    print(df['label'].value_counts())
