import pandas as pd
df = pd.read_csv("data/raw/creditcard.csv")
# Inject a clear pattern: If Amount > 1000 OR V1 is highly negative, make it FRAUD
df.loc[(df['Amount'] > 1000) | (df['V1'] < -3.0), 'Class'] = 1
df.to_csv("data/raw/creditcard.csv", index=False)
print("Pattern injected!")
