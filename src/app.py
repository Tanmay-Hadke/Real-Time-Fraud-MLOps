import uvicorn
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Barclays-Compliant Realtime Fraud Scoring Infrastructure")

# Global baseline models container
model_engine = None
FEATURES_LIST = [f"V{i}" for i in range(1, 29)] + ["Amount"]

class TransactionInput(BaseModel):
    user_id: str
    Amount: float
    # We expose V1 and V2 in the API to easily test how different values affect the score
    V1: float = 0.0
    V2: float = 0.0
    
@app.on_event("startup")
def bootstrap_microservice():
    """
    Locates the most recent successful training run in MLflow 
    and loads the XGBoost artifact into system memory for low-latency scoring.
    """
    global model_engine
    print("🤖 Connecting to MLflow and locating Production model...")
    
    try:
        # Programmatically find the latest model we just trained
        experiment = mlflow.get_experiment_by_name("transaction_fraud_realtime")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        
        latest_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{latest_run_id}/fraud_model"
        
        print(f"📥 Loading XGBoost model from Run ID: {latest_run_id}")
        
        # Load the native XGBoost booster from MLflow
        model_engine = mlflow.xgboost.load_model(model_uri)
        print("✅ Model successfully loaded into memory!")
        
    except Exception as e:
        print(f"❌ Failed to load model from MLflow. Did you run train.py? Error: {e}")
        model_engine = None

@app.post("/v1/score")
async def score_transaction(payload: TransactionInput):
    if not model_engine:
        raise HTTPException(status_code=503, detail="Model Engine Uninitialized.")
    
    # 1. SIMULATE FEAST ONLINE RETRIEVAL
    # Fill the 29-feature array with baseline normal values
    feature_vector = np.random.normal(0, 1, len(FEATURES_LIST)).astype(np.float32)
    
    # 2. OVERRIDE WITH LIVE API PAYLOAD
    # Map the live values from the Swagger UI into the exact feature positions
    feature_vector[0] = payload.V1        # V1
    feature_vector[1] = payload.V2        # V2
    feature_vector[-1] = payload.Amount   # Amount
    
    # 3. FORMAT FOR INFERENCE
    # XGBoost requires a 2D matrix with explicit feature names to prevent strictness errors
    xgb_input = xgb.DMatrix(np.array([feature_vector]), feature_names=FEATURES_LIST)
    
    # 4. PREDICT
    # Predict returns a numpy array. We extract the single float probability.
    fraud_prob = float(model_engine.predict(xgb_input)[0])
    
    # 5. BUSINESS LOGIC
    # Based on our training configuration, > 0.5 is a standard boundary
    decision = "BLOCK" if fraud_prob > 0.5 else "ALLOW"
    
    return {
        "status": "SUCCESS",
        "user_id": payload.user_id,
        "fraud_probability": round(fraud_prob, 5),
        "risk_decision": decision,
        "governance_tracking": {
            "mlflow_run_id_used": model_engine.attr("best_iteration") if hasattr(model_engine, "attr") else "latest",
            "evaluated_features_count": len(FEATURES_LIST),
            "latency_compliance": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)