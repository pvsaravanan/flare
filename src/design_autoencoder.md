# Deep Autoencoder Design for FLARE

## Architecture Specification

**Type**: Deep Undercomplete Autoencoder (U-AE)
**Input Dimension**: 13 (Numerical Flow Features)

### Layer-by-Layer Structure

| Layer Type          | Units / Shape | Activation    | Purpose                                    |
| :------------------ | :------------ | :------------ | :----------------------------------------- |
| **Input**           | 13            | None          | Normalized Input Vector `[0, 1]`           |
| **Encoder Dense 1** | 32            | ReLU          | Expansion/Feature Mixing                   |
| **Encoder Dense 2** | 16            | ReLU          | Compression                                |
| **Bottleneck**      | **8**         | None (Linear) | Latent Representation (Compressed Code)    |
| **Decoder Dense 1** | 16            | ReLU          | Reconstruction                             |
| **Decoder Dense 2** | 32            | ReLU          | Expansion                                  |
| **Output**          | 13            | **Sigmoid**   | Reconstructed Output (Matches Input Range) |

> **Note**: Output activation is `Sigmoid` because inputs are MinMax scaled to `[0, 1]`.

## Loss Function

**Mean Squared Error (MSE)**:
$$ L(x, \hat{x}) = \frac{1}{N} \sum (x_i - \hat{x}\_i)^2 $$

- Penalizes large deviations in reconstruction.
- Ideal for anomaly detection where we define Anomaly Score = MSE.

## Training Strategy (Federated)

1.  **Local Training**: Each client trains on **local normal traffic** (unsupervised).
2.  **Optimizer**: Adam (`lr=0.001`).
3.  **Federated Aggregation**: `FedAvg`. Averaging weights of encoders/decoders works well for AEs as they learn similar manifold representations of "normalcy".
4.  **Privacy**: Gradients/Weights are shared, but no raw flow data leaves the device.

## Anomaly Scoring

1.  **Inference**: Pass new sample $x$ through model to get $\hat{x}$.
2.  **Feature-wise Error**: $e = (x - \hat{x})^2$
3.  **Anomaly Score**: $MSE = mean(e)$
4.  **Thresholding**:
    - **Dynamic**: $Threshold = \mu + 3\sigma$ (of validation normal scores) or 95th percentile.
    - If $MSE > Threshold \rightarrow$ **Zero-Day Attack**.

## Suitability for Federated Environments

1.  **Compactness**: Small parameter count (~1-2k params) ensures minimal network overhead during FL rounds.
2.  **Unsupervised**: Clients do not need labelled attack data (which is rare). They only need "business as usual" traffic.
3.  **Robustness**: By averaging weights from multiple diverse IoT environments (Federated Averaging), the Global AE learns a more generalized definition of "Normal IoT Traffic", reducing false positives compared to isolated local models.
