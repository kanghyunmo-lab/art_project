import pandas as pd

# 파일 경로
train_path = r"c:/art_project/data/processed/btcusdt_labeled_features_train.parquet"
test_path = r"c:/art_project/data/processed/btcusdt_labeled_features_test.parquet"

# 데이터 로드
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 레이블 분포 출력
print("[Train] bin label distribution:")
print(train_df['bin'].value_counts(dropna=False))
print("\n[Test] bin label distribution:")
print(test_df['bin'].value_counts(dropna=False))
