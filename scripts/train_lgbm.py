import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 파일 경로
train_path = r"c:/art_project/data/processed/btcusdt_labeled_features_train.parquet"
test_path = r"c:/art_project/data/processed/btcusdt_labeled_features_test.parquet"

# 데이터 로드
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 레이블이 NaN이 아닌 구간만 추출
train_df = train_df[train_df['bin'].notna()]
test_df = test_df[test_df['bin'].notna()]

# 피처/타겟 분리 (OHLCV, 기술적지표, 4h 피처만 사용)
exclude_cols = ['bin', 'ret', 't_event_end']
X_train = train_df.drop(columns=[col for col in exclude_cols if col in train_df.columns])
y_train = train_df['bin'].astype(int)
X_test = test_df.drop(columns=[col for col in exclude_cols if col in test_df.columns])
y_test = test_df['bin'].astype(int)

# NaN이 포함된 행 제거
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

# train/test 컬럼 교집합만 사용
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# SMOTE로 클래스 불균형 보정
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# LightGBM 모델 학습
model = LGBMClassifier(random_state=42, n_estimators=300)
model.fit(X_train_res, y_train_res)

# 예측 및 평가
y_pred = model.predict(X_test)
print("[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))
