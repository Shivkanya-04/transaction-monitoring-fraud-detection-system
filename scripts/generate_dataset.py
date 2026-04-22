import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 1_000_000
start_date = datetime(2023, 1, 1)
data = {
    'transaction_id': range(1, n + 1),
    'user_id': np.random.randint(1, 50001, n),
    'amount': np.random.gamma(2, 50, n).astype(float),   # <-- float from beginning
    'timestamp': [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n)],
    'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'Unknown'], n, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05]),
    'device': np.random.choice(['mobile', 'desktop', 'tablet'], n, p=[0.6, 0.3, 0.1])
}

df = pd.DataFrame(data)

fraud_idx = np.random.choice(df.index, size=int(0.05 * n), replace=False)
df.loc[fraud_idx, 'amount'] = df.loc[fraud_idx, 'amount'] * np.random.uniform(3, 10, len(fraud_idx))
df.loc[fraud_idx, 'location'] = np.random.choice(['RU', 'NG', 'Unknown'], len(fraud_idx))
df.loc[fraud_idx, 'device'] = 'new_device'

df['is_fraud'] = 0
df.loc[fraud_idx, 'is_fraud'] = 1
df.to_csv('transactions.csv', index=False)
print(f"Dataset saved: {n:,} rows, {df['is_fraud'].sum():,} fraud cases ({df['is_fraud'].mean()*100:.1f}%)")
