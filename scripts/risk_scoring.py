import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("Step 4: Risk Scoring (Rules + ML)")

df = pd.read_csv('transactions.csv')
df_features = pd.read_csv('engineered_features.csv')
labels = pd.read_csv('labels.csv')

if 'is_fraud' in df.columns:
    df = df.drop(columns=['is_fraud'])

df = df.reset_index(drop=True)
df_features = df_features.reset_index(drop=True)
labels = labels.reset_index(drop=True)

df = pd.concat([df, df_features, labels[['is_fraud']]], axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Data loaded: {len(df):,} rows")

def rule_based_score(row):
    score = 0
    if row['amount'] > 5000: score += 30
    if row['geo_mismatch'] == 1: score += 25
    if row['velocity_1h'] > 5: score += 20
    if row['night_txn'] == 1: score += 10
    if row['device_mismatch'] == 1: score += 15
    return min(score, 100)

df['rule_score'] = df.apply(rule_based_score, axis=1)

feature_cols = [col for col in df_features.columns if col not in ['is_fraud']]
X = df[feature_cols].fillna(0)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

iso = IsolationForest(contamination=0.05, random_state=42)
iso_pred = iso.fit_predict(X)
df['iso_anomaly'] = (iso_pred == -1).astype(int)

xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred_prob = xgb.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"XGBoost AUC: {auc:.3f}")

df['xgb_prob'] = xgb.predict_proba(X)[:, 1]
df['ml_score'] = (df['iso_anomaly'] * 40 + df['xgb_prob'] * 60).clip(0, 1) * 100
df['risk_score'] = (0.6 * df['rule_score'] + 0.4 * df['ml_score']).astype(int)

original_max = df['risk_score'].max()
if original_max < 100:
    df['risk_score'] = (df['risk_score'] / original_max * 100).astype(int)
    print(f"Rescaled risk_score from max {original_max} to 100. New max: {df['risk_score'].max()}")

print(f"Risk score distribution: min={df['risk_score'].min()}, max={df['risk_score'].max()}, mean={df['risk_score'].mean():.1f}")
print(f"High-risk transactions (score > 80): {(df['risk_score'] > 80).sum():,}")

dashboard_cols = ['transaction_id', 'user_id', 'amount', 'timestamp', 'location', 'device', 'risk_score', 'is_fraud']
df[dashboard_cols].to_csv('dashboard_ready.csv', index=False)
print("Dashboard file saved: dashboard_ready.csv")

print("\nSample of high-risk transactions (risk_score > 80):")
high_risk = df[df['risk_score'] > 80].head(5)
print(high_risk[['transaction_id', 'amount', 'location', 'risk_score', 'is_fraud']])
