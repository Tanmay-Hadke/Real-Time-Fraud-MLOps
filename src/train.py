import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

def prepare_synthetic_feast_source(csv_path: str) -> pd.DataFrame:
    """
    Reads the synthetic raw CSV and ensures Feast-mandated timestamps
    are populated before exporting the baseline Parquet file.
    """
    print(f"📖 Reading local synthetic dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Feast requires explicit datetime objects to prevent data leakage.
    # If the bootstrap script didn't add them, we inject them here dynamically.
    if "event_timestamp" not in df.columns:
        print("⏱️ Injecting baseline timestamps for Feast timeline alignment...")
        base_time = pd.Timestamp.now() - pd.Timedelta(days=30)
        df["event_timestamp"] = base_time + pd.to_timedelta(df["Time"], unit="s")
        df["created_timestamp"] = pd.Timestamp.now()
        
    # Ensure types match Feast schema definitions exactly
    df["Class"] = df["Class"].astype(int)
    for i in range(1, 29):
        df[f"V{i}"] = df[f"V{i}"].astype(np.float32)
    df["Amount"] = df["Amount"].astype(np.float32)
    
    # Export to the Parquet source location that feature_repository/definitions.py relies on
    parquet_path = "data/baseline_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"💾 Synced Feast baseline data asset to {parquet_path}")
    return df

def run_training_pipeline():
    csv_path = "data/raw/creditcard.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ Synthetic data missing at {csv_path}. Please run your data bootstrap generation script first!"
        )
        
    # 1. Prepare and load the data
    df = prepare_synthetic_feast_source(csv_path)
    
    # Separate core feature matrices from target labels
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols]
    y = df["Class"]
    
    # Stratified split ensures fraud target ratios are preserved evenly across training sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Initialize Local MLflow Experiment Space
    mlflow.set_experiment("transaction_fraud_realtime")
    
    with mlflow.start_run() as run:
        print("🚀 Launching XGBoost training session tracked via MLflow...")
        
        # Calculate positive class scaling factor to combat extreme imbalance
        num_negatives = len(y_train) - sum(y_train)
        num_positives = sum(y_train)
        scale_pos_weight = num_negatives / (num_positives if num_positives > 0 else 1)
        
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",          # Changed from "prauc" to "aucpr"
            "scale_pos_weight": scale_pos_weight,
            "max_depth": 5,
            "learning_rate": 0.1,
            "tree_method": "hist"            
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Execute training iteration loop
        bst = xgb.train(
            params, 
            dtrain, 
            num_boost_round=30, 
            evals=[(dtest, "test")],
            verbose_eval=False
        )
        
        # 3. Performance Metrics Calculation
        preds = bst.predict(dtest)
        precision, recall, _ = precision_recall_curve(y_test, preds)
        pr_auc = auc(recall, precision)
        
        # Log architectural properties and results directly to local mlruns directory
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("synthetic_samples_count", len(df))
        
        # Register the binary inside our MLflow local model asset catalog
        mlflow.xgboost.log_model(
            xgb_model=bst, 
            artifact_path="fraud_model",
            registered_model_name="XGBFraudModelLocal"
        )
        
        print(f"\n🎉 Model successfully registered!")
        print(f"📈 Run ID: {run.info.run_id}")
        print(f"📊 Synthetic Test PR-AUC Score: {pr_auc:.4f}")
        print(f"🗂️ Run 'mlflow ui' in your terminal to inspect the dashboard tracking metrics.")

if __name__ == "__main__":
    run_training_pipeline()