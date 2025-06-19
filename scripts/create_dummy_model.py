#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
더미 머신러닝 모델을 생성하는 스크립트
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def create_dummy_model():
    # 모델 저장 디렉토리 생성
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 더미 특성과 레이블 생성 (100개 샘플, 5개 특성)
    np.random.seed(42)  # 재현성을 위한 시드 설정
    dummy_features = np.random.rand(100, 5)
    dummy_labels = np.random.choice([-1, 0, 1], size=100)
    
    # 모델 생성 및 훈련
    print("더미 모델을 생성하고 있습니다...")
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    dummy_model.fit(dummy_features, dummy_labels)
    
    # 모델 저장
    model_path = 'models/model_v1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(dummy_model, f)
    
    print(f"\n더미 모델이 성공적으로 생성되어 '{model_path}'에 저장되었습니다.")
    print("\n모델 정보:")
    print(f"- 클래스: {dummy_model.__class__.__name__}")
    print(f"- 특성 개수: {dummy_model.n_features_in_}")
    print(f"- 클래스 레이블: {dummy_model.classes_}")

if __name__ == '__main__':
    create_dummy_model()
