from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, df: pd.DataFrame, target_col='label'):
        """Fits scaler on data and transforms it. Returns X (scaled) and y."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values

    def transform(self, df: pd.DataFrame, target_col='label'):
        """Transforms new data using fitted scaler."""
        X = df.drop(columns=[target_col])
        y = df[target_col] if target_col in df.columns else None
        
        X_scaled = self.scaler.transform(X)
        return X_scaled, y.values if y is not None else None
        
    def save(self, path="c:/proj/flare/src/client/scaler.pkl"):
        """Saves the scaler to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load(self, path="c:/proj/flare/src/client/scaler.pkl"):
        """Loads the scaler from a file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler not found at {path}")

def get_dataloaders(X, y, batch_size=32, test_split=0.2):
    """Creates PyTorch DataLoaders."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    # Convert to Tensor
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
