# FLARE Client Pipeline Algorithm

This document defines the pseudo-algorithm for the Local Training and Inference Pipeline on a FLARE Edge Client.

## 1. Data Preprocessing

**Input**: Raw Network Flow Records $D_{raw}$ (DataFrame)  
**Output**: Normalized Feature Matrix $X$

```python
FUNCTION Preprocess(D_raw):
    # 1. Feature Selection
    # Select 13 numerical features (Duration, Throughput, TCP Flags, IAT stats)
    Features = D_raw.select([
        'flow_duration', 'flow_bytes_s', 'flow_pkts_s',
        'flow_iat_mean', 'flow_iat_std',
        'fwd_pkt_len_max', 'total_fwd_pkts', 'total_bwd_pkts',
        'syn_flag_count', 'rst_flag_count', 'fin_flag_count',
        'active_mean', 'idle_mean'
    ])

    # 2. Cleaning
    # Clip negative values to 0
    Features = Features.clip(lower=0)
    # Handle infinite values (replace with max finite or 0)
    Features = Features.replace(Infinity, 0)

    # 3. Normalization (MinMax Scaling)
    # Critical for Autoencoder with Sigmoid Output
    # Scaler is pre-fitted on a representative 'Normal' baseline or updated incrementally
    X = Scaler.transform(Features) -> Range [0, 1]

    RETURN X
```

## 2. Local Training Loop

**Input**: Local Normal Data $X_{train}$ (Label=0), Global Model Weights $W_{global}$  
**Output**: Updated Model Weights $W_{local}$

```python
FUNCTION LocalSearch(X_train, W_global, Epochs, BatchSize, LearningRate):
    # 1. Initialize Local Model
    Model = DeepAutoencoder()
    Model.load_weights(W_global)

    Optimizer = Adam(LearningRate)
    Criterion = MSELoss()

    # 2. Training Loop
    FOR epoch in 1 to Epochs:
        # Shuffle Data
        Batches = Split(X_train, BatchSize)

        FOR batch in Batches:
            # Forward Pass
            Reconstruction = Model.forward(batch)

            # Compute Loss
            Loss = Criterion(batch, Reconstruction)

            # Backward Pass
            Gradients = ComputeGradients(Loss)
            Optimizer.update_weights(Model, Gradients)

    # 3. Extract Weights
    W_local = Model.get_weights()

    RETURN W_local
```

## 3. Anomaly Detection Logic

**Input**: New Traffic Sample $x_{new}$, Trained Model $M$, Threshold $\tau$  
**Output**: IsAnomaly (Boolean), Score (Float)

```python
FUNCTION DetectAnomaly(x_new, M, tau):
    # 1. Preprocess
    x_scaled = Preprocess(x_new)

    # 2. Inference
    with NoGrad():
        x_recon = M.forward(x_scaled)

    # 3. Calculate Reconstruction Error (MSE)
    # Error vector e = (x - x_recon)^2
    ErrorVector = (x_scaled - x_recon)^2
    AnomalyScore = Mean(ErrorVector) # Scalar MSE

    # 4. Decision
    IF AnomalyScore > tau:
        RETURN True, AnomalyScore
    ELSE:
        RETURN False, AnomalyScore
```

## 4. Threshold Selection Strategy

**Input**: Validation Set of Normal Traffic $X_{val}$  
**Output**: Threshold $\tau$

```python
FUNCTION SelectThreshold(X_val, M):
    Scores = []

    # 1. Calculate Scores for all validation samples
    FOR x in X_val:
        _, score = DetectAnomaly(x, M, 0) # Threshold irrelevant here
        Scores.append(score)

    # 2. Statistical Thresholding
    # Option A: Percentile (Robust to outliers)
    tau = Percentile(Scores, 95) # 95th percentile of normal reconstruction error

    # Option B: Gaussian assumption (Mean + 3*Std)
    # tau = Mean(Scores) + 3 * StdDev(Scores)

    RETURN tau
```
