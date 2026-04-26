import pandas as pd
import glob
import os

# Path to results directory
results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
pattern = os.path.join(results_dir, 'baseline*.csv')

for csv_path in glob.glob(pattern):
    df = pd.read_csv(csv_path)
    if 'predicted_score' in df.columns and 'predicted_label' in df.columns:
        df['predicted_label'] = df['predicted_score'].apply(lambda x: 1 if x >= 70 else 0)
        df.to_csv(csv_path, index=False)
        print(f"Updated: {csv_path}")
    else:
        print(f"Skipped (missing columns): {csv_path}")
