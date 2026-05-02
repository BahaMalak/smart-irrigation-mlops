import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib
import os

# ══════════════════════════════════════════
# 1. CHARGER ET MERGER LES 3 FICHIERS
# ══════════════════════════════════════════
print(" Chargement des données...")

df1 = pd.read_csv("data/raw/plant_vase1.CSV")
df2 = pd.read_csv("data/raw/plant_vase1(2).CSV")
df3 = pd.read_csv("data/raw/plant_vase2.CSV")

df = pd.concat([df1, df2, df3], ignore_index=True)
print(f" Dataset total : {df.shape[0]} lignes")

# ══════════════════════════════════════════
# 2. CRÉER LA TARGET INTELLIGEMMENT
# ══════════════════════════════════════════
# Moyenne des 5 capteurs
df["moisture_mean"] = df[["moisture0","moisture1",
                           "moisture2","moisture3",
                           "moisture4"]].mean(axis=1)

# Si humidité moyenne < 0.35 → besoin d'irrigation (1)
# Sinon → pas besoin (0)
THRESHOLD = 0.35
df["irrigation"] = (df["moisture_mean"] < THRESHOLD).astype(int)

print(f"\n Distribution target (seuil={THRESHOLD}):")
print(df["irrigation"].value_counts())
print(f"Pourcentage irrigation needed : {df['irrigation'].mean():.1%}")

# ══════════════════════════════════════════
# 3. FEATURES + SPLIT
# ══════════════════════════════════════════
features = ["moisture0", "moisture1", "moisture2",
            "moisture3", "moisture4", "hour"]

X = df[features]
y = df["irrigation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Scaler pour Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ══════════════════════════════════════════
# 4. LES 3 MODÈLES + MLFLOW
# ══════════════════════════════════════════
models = {
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

mlflow.set_experiment("smart-irrigation")

best_model    = None
best_accuracy = 0
best_name     = ""

print("\n Entraînement des modèles...\n")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Entraînement
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Métriques
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)

        # Logger dans MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_param("threshold",  THRESHOLD)
        mlflow.log_param("features",   str(features))
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.sklearn.log_model(model, name)

        print(f" {name:<22} Accuracy: {acc:.3f} | F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")

        # Garder le meilleur
        if acc > best_accuracy:
            best_accuracy = acc
            best_model    = model
            best_name     = name

# ══════════════════════════════════════════
# 5. SAUVEGARDER LE MEILLEUR MODÈLE
# ══════════════════════════════════════════
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(features,   "models/features.pkl")

print(f"\n Meilleur modèle : {best_name} (Accuracy: {best_accuracy:.3f})")
print(" Modèle sauvegardé dans models/best_model.pkl")
