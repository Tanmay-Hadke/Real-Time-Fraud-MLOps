import numpy as np

def calculate_feature_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between a training baseline 
    and incoming operational transaction arrays.
    
    Threshold Guidelines:
    - PSI < 0.1: Stable; no action required.
    - 0.1 <= PSI < 0.25: Marginal drift detected; queue monitoring.
    - PSI >= 0.25: Critical drift detected; action retraining pipeline.
    """
    # Create identical structural quantile buckets using baseline expectations
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    # Quantify data density inside matching buckets
    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)
    
    # Convert to probability distributions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Clean zero frequencies with Laplace-style smoothing constants to avoid mathematical zero division
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Complete PSI algorithm logic execution
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

if __name__ == "__main__":
    # Simulate stable and drifted distributions for local testing verification
    baseline = np.random.normal(0, 1, 5000)
    live_stable = np.random.normal(0.02, 1, 5000)
    live_drifted = np.random.normal(0.45, 1, 5000)
    
    print(f"Stable Distribution PSI Check: {calculate_feature_psi(baseline, live_stable):.4f}")
    print(f"Drifted Distribution PSI Check: {calculate_feature_psi(baseline, live_drifted):.4f}")