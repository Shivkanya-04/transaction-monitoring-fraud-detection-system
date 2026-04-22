import pandas as pd
import numpy as np


df = pd.read_csv('transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['user_id', 'timestamp'])

print(f"Loaded {len(df):,} transactions")

# ----- Feature 1: Transaction velocity -----
df['timestamp_sec'] = df['timestamp'].astype('int64') // 10**9  # convert to Unix seconds
def rolling_count_1h(group):
    group = group.sort_values('timestamp_sec')
    counts = []
    for i, row in group.iterrows():
        window_start = row['timestamp_sec'] - 3600
        count = group[(group['timestamp_sec'] >= window_start) & (group['timestamp_sec'] < row['timestamp_sec'])].shape[0]
        counts.append(count)
    group['velocity_1h'] = counts
    return group

df = df.set_index('timestamp')
df['velocity_1h'] = df.groupby('user_id')['amount'].rolling('1H').count().values
df = df.reset_index()
df['velocity_1h'] = df['velocity_1h'] - 1  
df['velocity_1h'] = df['velocity_1h'].fillna(0).astype(int)

user_avg = df.groupby('user_id')['amount'].transform('mean')
user_std = df.groupby('user_id')['amount'].transform('std')
df['spend_deviation'] = (df['amount'] - user_avg) / user_std.fillna(1)
df['spend_deviation'] = df['spend_deviation'].fillna(0).clip(-5, 5)

user_main_loc = df.groupby('user_id')['location'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
df['geo_mismatch'] = (df['location'] != df['user_id'].map(user_main_loc)).astype(int)

df['hour'] = df['timestamp'].dt.hour
df['night_txn'] = ((df['hour'] >= 0) & (df['hour'] < 5)).astype(int)

user_main_device = df.groupby('user_id')['device'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'mobile')
df['device_mismatch'] = (df['device'] != df['user_id'].map(user_main_device)).astype(int)
df['amount_log'] = np.log1p(df['amount'])

df['txn_hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['txn_hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['txn_count_today'] = df.groupby(['user_id', df['timestamp'].dt.date])['transaction_id'].transform('count')
df['amount_percentile_user'] = df.groupby('user_id')['amount'].rank(pct=True)
df['time_since_last_txn_hr'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
df['time_since_last_txn_hr'] = df['time_since_last_txn_hr'].fillna(0)

feature_cols = [
    'velocity_1h', 'spend_deviation', 'geo_mismatch', 'night_txn',
    'device_mismatch', 'amount_log', 'txn_hour_sin', 'txn_hour_cos',
    'day_of_week', 'is_weekend', 'txn_count_today', 'amount_percentile_user',
    'time_since_last_txn_hr'
]

df_features = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
df_features.to_csv('engineered_features.csv', index=False)
df[['transaction_id', 'is_fraud']].to_csv('labels.csv', index=False)

print(f"Feature engineering completed. Shape: {df_features.shape}")
print(f"Features created: {len(feature_cols)} (you can add more to reach 50+)")
print("Files saved: engineered_features.csv, labels.csv")
