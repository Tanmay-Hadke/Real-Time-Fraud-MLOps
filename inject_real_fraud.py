import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/creditcard.csv")

# Generate 200 explicitly fraudulent rows
n_fraud = 200
fraud_data = pd.DataFrame({
    "Time": np.random.uniform(0, 86400, n_fraud),
    "Amount": np.random.uniform(5000, 10000, n_fraud),  # Massive transactions
    "Class": np.ones(n_fraud, dtype=int),               # Explicitly mark as fraud
    "user_id": [f"USER_HACKER_{i}" for i in range(n_fraud)]
})

# Inject extreme V1 values for these fraud rows
for i in range(1, 29):
    if i == 1:
        fraud_data[f"V{i}"] = np.random.uniform(-10, -5, n_fraud)
    else:
        fraud_data[f"V{i}"] = np.random.normal(0, 1, n_fraud)

# Append to original data and save
fraud_data = fraud_data[df.columns]
df = pd.concat([df, fraud_data], ignore_index=True)

df.to_csv("data/raw/creditcard.csv", index=False)
print("Explicit extreme fraud rows appended to dataset!")
