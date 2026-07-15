import streamlit as st
import pandas as pd
import numpy as np

# Configure the Streamlit page layout
st.set_page_config(page_title="MLOps Governance Monitor", page_icon="🏦", layout="wide")

@st.cache_data
def load_baseline_data():
    """Loads the historical training baseline from the Feast offline store."""
    try:
        return pd.read_parquet("data/baseline_features.parquet")
    except FileNotFoundError:
        st.error("Baseline data not found. Please ensure data/baseline_features.parquet exists.")
        st.stop()

def calculate_psi(expected, actual, num_buckets=10):
    """Calculates the Population Stability Index for a given feature."""
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0], buckets[-1] = -np.inf, np.inf
    
    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def generate_live_traffic_simulation(baseline_df, drift_intensity=0.0):
    """
    Simulates the last 24 hours of incoming API traffic.
    drift_intensity allows us to artificially skew the data to test our alerts.
    """
    live_df = baseline_df.sample(n=1000, replace=True).copy()
    
    # Artificially inject drift into the Amount feature if requested
    if drift_intensity > 0:
        live_df['Amount'] = live_df['Amount'] * (1 + drift_intensity)
        
    return live_df

# --- UI LAYOUT ---

st.title("🏦 Production MLOps & Governance Dashboard")
st.markdown("Real-time monitoring for the Fraud Detection Scoring Engine.")
st.divider()

# Load foundational data
baseline_df = load_baseline_data()

# Sidebar Controls for Simulation
st.sidebar.header("Simulate Production Environment")
st.sidebar.markdown("Use this slider to introduce mathematical drift into the live transaction stream.")
drift_factor = st.sidebar.slider("Inject 'Amount' Feature Drift", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# Generate the mock live traffic window
live_df = generate_live_traffic_simulation(baseline_df, drift_intensity=drift_factor)

# Calculate PSI for the critical 'Amount' feature
psi_score = calculate_psi(baseline_df['Amount'].values, live_df['Amount'].values)

# --- TOP LEVEL KPIs ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Active Model Version", value="v1.0 (XGBoost)")
with col2:
    st.metric(label="Transactions (Last 24h)", value=f"{len(live_df):,}")
with col3:
    fraud_rate = (live_df['Class'].sum() / len(live_df)) * 100
    st.metric(label="Current Fraud Rate", value=f"{fraud_rate:.2f}%")
with col4:
    # Color-code the PSI metric based on severity thresholds
    if psi_score < 0.1:
        st.metric(label="Amount PSI (Drift)", value=f"{psi_score:.4f}", delta="Stable", delta_color="normal")
    elif psi_score < 0.25:
        st.metric(label="Amount PSI (Drift)", value=f"{psi_score:.4f}", delta="Warning", delta_color="off")
    else:
        st.metric(label="Amount PSI (Drift)", value=f"{psi_score:.4f}", delta="CRITICAL", delta_color="inverse")

st.divider()

# --- ALERTING LOGIC ---
if psi_score >= 0.25:
    st.error(f"🚨 **CRITICAL ALERT:** Severe data drift detected in feature 'Amount' (PSI: {psi_score:.4f}). Automated Airflow retraining pipeline triggered.")
elif psi_score >= 0.1:
    st.warning(f"⚠️ **WARNING:** Marginal data drift detected in feature 'Amount' (PSI: {psi_score:.4f}). Monitor closely.")
else:
    st.success(f"✅ **SYSTEM STABLE:** Feature distributions are aligned with the training baseline (PSI: {psi_score:.4f}).")

# --- VISUALIZATION ---
st.subheader("Feature Distribution Comparison: Baseline vs. Live Traffic")

# Prepare data for plotting
# We use log scale because transaction amounts are highly skewed
plot_data = pd.DataFrame({
    'Baseline (Training)': np.log1p(baseline_df['Amount'].sample(1000).values),
    'Live (Last 24h)': np.log1p(live_df['Amount'].values)
})

st.area_chart(plot_data, color=["#1f77b4", "#d62728"])

st.caption("Visualizing the log-transformed distribution of transaction amounts. Overlap indicates stability; divergence indicates drift.")