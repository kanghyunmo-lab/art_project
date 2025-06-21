#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models.training 모듈의 테스트 코드
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 테스트할 모듈 가져오기
from models.training import XGBoostModel
from models.base import BaseLogger


class TestXGBoostModel(unittest.TestCase):
    """XGBoostModel 클래스의 단위 테스트를 위한 테스트 케이스"""

    def setUp(self):
        """각 테스트 메서드 실행 전에 호출됨"""
        # 테스트용 로거 설정
        self.logger = BaseLogger("TestLogger", log_dir=os.path.join(project_root, 'tests', 'logs'))
        
        # 테스트용 모델 파라미터
        self.model_params = {
            'n_estimators': 10,  # 빠른 테스트를 위해 작은 값 사용
            'max_depth': 2,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42
        }
        
        # XGBoostModel 인스턴스 생성
        self.model = XGBoostModel(self.model_params, logger=self.logger)
        
        # 테스트용 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        # 피처 데이터
        self.X = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples) for i in range(10)
        })
        
        # 타겟 데이터 (3개 클래스)
        self.y = pd.Series(np.random.randint(0, 3, n_samples))
        
        # 테스트 디렉토리 설정
        self.test_dir = os.path.join(project_root, 'tests', 'models')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 임시 모델 파일 경로
        self.model_filepath = os.path.join(self.test_dir, 'test_model.joblib')
    
    def tearDown(self):
        """각 테스트 메서드 실행 후에 호출됨"""
        # 로거 핸들러 정리
        if self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
        
        # 테스트 파일 정리
        if os.path.exists(self.model_filepath):
            try:
                os.remove(self.model_filepath)
            except PermissionError:
                pass
        
        # features.txt 파일도 정리
        features_path = os.path.join(
            os.path.dirname(self.model_filepath),
            f"{os.path.splitext(os.path.basename(self.model_filepath))[0]}_features.txt"
        )
        if os.path.exists(features_path):
            try:
                os.remove(features_path)
            except PermissionError:
                pass
    
    def test_initialization(self):
        """초기화 테스트"""
        # 기본 파라미터 확인
        self.assertEqual(self.model.params['n_estimators'], 10)
        self.assertEqual(self.model.params['max_depth'], 2)
        self.assertEqual(self.model.params['learning_rate'], 0.1)
        self.assertEqual(self.model.params['num_class'], 3)
        
        # 기본값 사용 테스트
        default_model = XGBoostModel()
        self.assertIsNotNone(default_model.params)
        self.assertEqual(default_model.params['n_estimators'], 100)  # 기본값 확인
    
    def test_train_predict(self):
        """학습 및 예측 테스트"""
        try:
            # 모델 학습
            trained_model = self.model.train(self.X, self.y)
            self.assertIsNotNone(trained_model)
            self.assertIsNotNone(self.model.model)
            
            # 피처 이름 저장 확인
            self.assertEqual(set(self.model.feature_names), set(self.X.columns))
            
            # 예측 테스트
            y_pred = self.model.predict(self.X)
            self.assertEqual(len(y_pred), len(self.y))
            self.assertTrue(all(0 <= y <= 2 for y in y_pred))  # 클래스 범위 확인
            
            # 확률 예측 테스트
            y_prob = self.model.predict_proba(self.X)
            self.assertEqual(y_prob.shape, (len(self.y), 3))  # 3개 클래스에 대한 확률
            
        except ModuleNotFoundError:
            self.skipTest("xgboost가 설치되지 않아 테스트를 건너뜁니다.")
    
    def test_save_load(self):
        """모델 저장 및 로드 테스트"""
        try:
            # 모델 학습
            self.model.train(self.X, self.y)
            
            # 모델 저장
            self.model.save(self.model_filepath)
            self.assertTrue(os.path.exists(self.model_filepath))
            
            # 특성 이름 파일 저장 확인
            feature_names_path = os.path.join(
                os.path.dirname(self.model_filepath),
                f"{os.path.splitext(os.path.basename(self.model_filepath))[0]}_features.txt"
            )
            self.assertTrue(os.path.exists(feature_names_path))
            
            # 모델 로드
            loaded_model = XGBoostModel.load(self.model_filepath)
            self.assertIsNotNone(loaded_model.model)
            
            # 로드된 모델로 예측 테스트
            y_pred_original = self.model.predict(self.X)
            y_pred_loaded = loaded_model.predict(self.X)
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
            
            # 로드된 모델의 특성 이름 확인
            self.assertEqual(set(loaded_model.feature_names), set(self.X.columns))
            
        except ModuleNotFoundError:
            self.skipTest("xgboost가 설치되지 않아 테스트를 건너뜁니다.")
    
    def test_feature_importance(self):
        """특성 중요도 추출 테스트"""
        try:
            # 모델 학습
            self.model.train(self.X, self.y)
            
            # 특성 중요도 추출
            importance_df = self.model.get_feature_importance()
            
            # 데이터프레임 검증
            self.assertIsInstance(importance_df, pd.DataFrame)
            self.assertEqual(len(importance_df), len(self.X.columns))
            self.assertTrue('feature' in importance_df.columns)
            self.assertTrue('importance' in importance_df.columns)
            
            # 모든 특성이 포함되어 있는지 확인
            self.assertEqual(set(importance_df['feature']), set(self.X.columns))
            
        except ModuleNotFoundError:
            self.skipTest("xgboost가 설치되지 않아 테스트를 건너뜁니다.")


if __name__ == '__main__':
    unittest.main()
