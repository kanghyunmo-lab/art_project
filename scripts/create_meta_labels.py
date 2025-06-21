# -*- coding: utf-8 -*-
"""
2단계 모델링용 레이블 생성 및 분포 확인 스크립트
- meta_label(거래 여부)
- action_df(매수/매도 데이터)
"""
import pandas as pd
import os

# 데이터 경로
train_path = r"c:/art_project/data/processed/btcusdt_labeled_features_train.parquet"
test_path = r"c:/art_project/data/processed/btcusdt_labeled_features_test.parquet"

# 데이터 로드
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# meta_label 생성: 0(보유) → 0, -1/1(매도/매수) → 1
train_df['meta_label'] = train_df['bin'].apply(lambda x: 0 if x == 0 else 1)
test_df['meta_label'] = test_df['bin'].apply(lambda x: 0 if x == 0 else 1)

print("[Train] Meta Label Distribution (0: No Action, 1: Action):")
print(train_df['meta_label'].value_counts())
print("\n[Test] Meta Label Distribution (0: No Action, 1: Action):")
print(test_df['meta_label'].value_counts())

# 거래 발생 데이터만 추출 (action_df)
train_action_df = train_df[train_df['meta_label'] == 1].copy()
test_action_df = test_df[test_df['meta_label'] == 1].copy()

print("\n[Train] Buy/Sell Distribution in Action Data:")
print(train_action_df['bin'].value_counts())
print("\n[Test] Buy/Sell Distribution in Action Data:")
print(test_action_df['bin'].value_counts())

# 필요시 parquet로 저장
train_df.to_parquet("c:/art_project/data/processed/btcusdt_labeled_features_train_with_meta.parquet")
test_df.to_parquet("c:/art_project/data/processed/btcusdt_labeled_features_test_with_meta.parquet")
train_action_df.to_parquet("c:/art_project/data/processed/btcusdt_action_train.parquet")
test_action_df.to_parquet("c:/art_project/data/processed/btcusdt_action_test.parquet")
