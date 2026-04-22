import sqlite3
import pandas as pd

print("Step 2: SQL Data Cleaning")

# Connect to (or create) SQLite database file
conn = sqlite3.connect('fraud.db')

# Load CSV into SQL table 'transactions'
print("Loading CSV into SQLite...")
df = pd.read_csv('transactions.csv')
df.to_sql('transactions', conn, if_exists='replace', index=False)
print("Table 'transactions' created.")

# Run cleaning queries
cursor = conn.cursor()

# 1. Remove duplicate transaction_ids (if any)
cursor.execute("""
    DELETE FROM transactions 
    WHERE transaction_id NOT IN (
        SELECT MIN(transaction_id) 
        FROM transactions 
        GROUP BY transaction_id
    )
""")
print(f"Duplicates removed. Rows left: {cursor.rowcount} (but rowcount may show -1; ignore)")

# 2. Remove negative amounts (should be none, but safe)
cursor.execute("DELETE FROM transactions WHERE amount < 0")
print("Negative amounts removed.")

# 3. Create aggregated view: daily transaction velocity per user
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_daily_velocity AS
    SELECT 
        user_id,
        DATE(timestamp) as tx_date,
        COUNT(*) as daily_tx_count,
        SUM(amount) as daily_volume
    FROM transactions
    GROUP BY user_id, DATE(timestamp)
""")
print("Table 'user_daily_velocity' created with daily aggregates.")

# Verify
print("\nPreview of user_daily_velocity:")
print(pd.read_sql("SELECT * FROM user_daily_velocity LIMIT 5", conn))

conn.commit()
conn.close()
print("\nSQL cleaning and aggregation completed. Database file: fraud.db")
