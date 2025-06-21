import pandas as pd

df = pd.read_parquet('data/processed/btcusdt_feature_matrix_test.parquet')
print('Test feature matrix shape:', df.shape)
nan_counts = df.isna().sum()
print('--- NaN count by column (descending) ---')
print(nan_counts[nan_counts > 0].sort_values(ascending=False))
print('--- Rows with any NaN:', df.isna().any(axis=1).sum())
print('--- First rows with NaN ---')
print(df[df.isna().any(axis=1)].head(10))
