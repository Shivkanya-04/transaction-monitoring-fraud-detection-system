import subprocess
import os
os.chdir(r'C:\Users\ThinkPad\Desktop\fraud_project')

scripts = [
    'generate_dataset.py',
    'clean_sql.py',
    'feature_engineering.py',
    'risk_scoring.py'
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run(['python', script], check=True)

print("\nFull pipeline completed.")
