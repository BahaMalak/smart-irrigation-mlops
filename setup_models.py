import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

print(" Creating models...")

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'moisture0': np.random.uniform(0.02, 0.85, n),
    'moisture1': np.random.uniform(0.14, 0.96, n),
    'moisture2': np.random.uniform(0.23, 0.98, n),
    'moisture3': np.random.uniform(0.08, 0.99, n),
    'moisture4': np.random.uniform(0.00, 0.11, n),
    'hour':      np.random.randint(0, 23, n)
})

df["moisture_mean"] = df[["moisture0","moisture1",
                           "moisture2","moisture3",
                           "moisture4"]].mean(axis=1)
df["irrigation"] = (df["moisture_mean"] < 0.35).astype(int)

features = ["moisture0","moisture1","moisture2",
            "moisture3","moisture4","hour"]

X = df[features]
y = df["irrigation"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model,    "models/best_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")

print(" Models created successfully!")