import pandas as pd
cols = pd.read_parquet('data/processed/btcusdt_labeled_features_test.parquet').columns
print('labeled_features_test:', list(cols))
