# ==========================================
# FRAUD DETECTION USING SYNTHETIC DATA
# ==========================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

# ------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ------------------------------------------

np.random.seed(42)

N_TRANSACTIONS = 12000
N_USERS = 1000
N_MERCHANTS = 200
FRAUD_RATE = 0.03

users = [f"user_{i}" for i in range(N_USERS)]
merchants = [f"merchant_{i}" for i in range(N_MERCHANTS)]
cities = ["NY", "LA", "London", "Delhi", "Berlin"]
payment_methods = ["card", "upi", "wallet", "netbanking"]

start_time = datetime.now()

records = []

for i in range(N_TRANSACTIONS):
    user = np.random.choice(users)
    merchant = np.random.choice(merchants)
    amount = np.random.exponential(scale=50)
    timestamp = start_time + timedelta(minutes=i)
    location = np.random.choice(cities)
    payment_method = np.random.choice(payment_methods)
    device_id = f"device_{np.random.randint(1, 300)}"

    is_fraud = 0

    # Fraud Pattern 1: High amount anomaly
    if amount > 300:
        is_fraud = 1

    # Fraud Pattern 2: Shared devices
    if device_id.endswith("7"):
        is_fraud = 1

    records.append([
        f"tx_{i}",
        user,
        merchant,
        round(amount, 2),
        timestamp,
        location,
        payment_method,
        device_id,
        is_fraud
    ])

df = pd.DataFrame(records, columns=[
    "transaction_id",
    "user_id",
    "merchant_id",
    "amount",
    "timestamp",
    "location",
    "payment_method",
    "device_id",
    "is_fraud"
])

# enforce fraud imbalance
df.loc[df.sample(frac=1 - FRAUD_RATE, random_state=42).index, "is_fraud"] = 0

print("Dataset shape:", df.shape)
print("Fraud rate:", df["is_fraud"].mean())

# ------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------

df = df.sort_values("timestamp")

# transaction count per user
df["user_tx_count"] = df.groupby("user_id").cumcount() + 1

# average transaction amount per user
df["user_avg_amount"] = df.groupby("user_id")["amount"].transform("mean")

# deviation from user average
df["amount_deviation"] = df["amount"] - df["user_avg_amount"]

# number of transactions per device
df["device_tx_count"] = df.groupby("device_id").cumcount() + 1

# one-hot encoding
df_encoded = pd.get_dummies(
    df,
    columns=["location", "payment_method"],
    drop_first=True
)

# ------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------

X = df_encoded.drop(columns=[
    "transaction_id",
    "user_id",
    "merchant_id",
    "device_id",
    "timestamp",
    "is_fraud"
])

y = df_encoded["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# ------------------------------------------
# 4. MODEL TRAINING
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------
# 5. EVALUATION
# ------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------
# 6. EXPLAINABILITY
# ------------------------------------------

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Important Fraud Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
