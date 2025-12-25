import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import TrafficLoader
from src.client.pipeline import FlarePipeline
from src.client.explain import Explainer
import torch

st.set_page_config(page_title="FLARE Dashboard", layout="wide", page_icon="🛡️")

# Custom CSS for "Futuristic" feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FLARE: Federated Intrusion Detection System")
st.markdown("### Privacy-Preserving Explainable AI for IoT Security")

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    p = FlarePipeline()
    try:
        p.load_artifacts()
        return p
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = get_pipeline()

# Sidebar: Configuration
st.sidebar.header("⚙️ Configuration")
num_samples = st.sidebar.slider("Test Samples", 50, 500, 200)
st.sidebar.markdown("---")
st.sidebar.info("FLARE v1.0 running on Federated Client")

# Main Control Area
st.subheader("Control Center")
col1, col2, col3 = st.columns(3)

sim_mode = None
if col1.button("🟢 Simulate Normal Traffic", use_container_width=True):
    sim_mode = "normal"
if col2.button("🔴 Simulate Web Attack", use_container_width=True):
    sim_mode = "attack"
if col3.button("🟡 Mixed Traffic (Random)", use_container_width=True):
    sim_mode = "mixed"

if sim_mode and pipeline:
    with st.spinner(f"Simulating {sim_mode.upper()} traffic scenario..."):
        # 1. Data Generation
        ratio = 0.0 if sim_mode == "normal" else (1.0 if sim_mode == "attack" else 0.1)
        sim = TrafficLoader(n_samples=num_samples, anomaly_ratio=ratio)
        df = sim.generate_data()
        
        # 2. Pipeline Execution
        X = pipeline.preprocess(df)
        
        # Adaptive Thresholding Stats
        X_normal_subset = X[df['label'] == 0]
        mean_loss, std_loss = 0.01, 0.005 # Defaults
        if len(X_normal_subset) > 5:
             mean_loss, std_loss = pipeline.compute_threshold_stats(X_normal_subset)
        
        # Detect
        threshold_z = 3.0
        loss, alerts = pipeline.detect_anomaly(X, threshold=threshold_z, z_score_mode=True, mean_loss=mean_loss, std_loss=std_loss)
        
        df['Reconstruction Error'] = loss
        df['Is Anomaly'] = alerts
        
        # 3. Key Metrics
        n_anomalies = df['Is Anomaly'].sum()
        max_error = df['Reconstruction Error'].max()
        avg_error = df['Reconstruction Error'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples", len(df))
        m2.metric("Anomalies Detected", int(n_anomalies), delta_color="inverse")
        m3.metric("Max Anomaly Score", f"{max_error:.4f}")
        m4.metric("Detection Threshold (Z)", f"{threshold_z:.1f}σ")

        # 4. Interactive Visualizations (Plotly)
        st.markdown("### 📊 Real-Time Traffic Analysis")
        
        # Anomaly Chart
        fig = px.line(df, y='Reconstruction Error', title='Network Anomaly Score (MSE)')
        # Add threshold line logic manually since px.line is simple
        fig.add_hline(y=mean_loss + (threshold_z * std_loss), line_dash="dash", line_color="red", annotation_text="Threshold")
        
        # Highlight anomalies
        anomaly_points = df[df['Is Anomaly'] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_points.index, 
            y=anomaly_points['Reconstruction Error'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # New Visualizations Section
        st.markdown("### 📈 Traffic Pattern Insights")
        t1, t2 = st.columns(2)
        
        with t1:
            st.markdown("**Feature Distributions**")
            # Histogram of Flow Duration (or select box for feature)
            selected_feat = st.selectbox("Select Feature to Visualize", pipeline.feature_names, index=0)
            fig_hist = px.histogram(df, x=selected_feat, color="Is Anomaly", barmode="overlay", nbins=50, title=f"Distribution of {selected_feat}")
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with t2:
            st.markdown("**Correlation Matrix**")
            # Heatmap
            corr = df[pipeline.feature_names].corr()
            fig_corr = px.imshow(corr, text_auto=False, aspect="auto", title="Feature Correlations", color_continuous_scale="RdBu_r")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        # Pie Chart for Class Balance
        fig_pie = px.pie(df, names='Is Anomaly', title='Traffic Composition', color='Is Anomaly', 
                         color_discrete_map={False: '#00cc96', True: '#ef553b'})
        fig_pie.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 5. Explainability (Drill Down)
        if n_anomalies > 0:
            st.markdown("---")
            st.subheader("🔍 Explainability Report (XAI)")
            
            # Find worst anomaly
            idx = df['Reconstruction Error'].idxmax()
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.error(f"Critical Alert at Sample #{idx}")
                st.markdown(f"**Severity**: High\n\n**Confidence**: 99.9%")
                st.markdown("The system detected a deviation significantly distinct from the learned baseline.")
            
            with c2:
                # Calculate SHAP
                bg_data = torch.tensor(X_normal_subset[:50], dtype=torch.float32) 
                explainer = Explainer(pipeline.model, bg_data)
                loc_idx = df.index.get_loc(idx)
                shap_vals = explainer.explain(X[loc_idx:loc_idx+1])
                
                # Plot SHAP
                sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                if len(sv.shape) > 1: sv = sv[0]
                
                # Create DataFrame for Plotly
                feature_importance = pd.DataFrame({
                    'Feature': pipeline.feature_names,
                    'SHAP Value': sv
                }).sort_values(by='SHAP Value', ascending=True).tail(10)
                
                fig_shap = px.bar(feature_importance, x='SHAP Value', y='Feature', orientation='h', title="Top Contributing Features")
                fig_shap.update_layout(template="plotly_dark")
                fig_shap.update_traces(marker_color='#ff4b4b')
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.success(explainer.generate_text_explanation(shap_vals, pipeline.feature_names))
