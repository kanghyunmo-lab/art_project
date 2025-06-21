#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models.evaluation 모듈의 테스트 코드
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.metrics import confusion_matrix

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 테스트할 모듈 가져오기
from models.evaluation import ModelEvaluator
from models.base import BaseLogger


class TestModelEvaluator(unittest.TestCase):
    """ModelEvaluator 클래스의 단위 테스트를 위한 테스트 케이스"""

    def setUp(self):
        """각 테스트 메서드 실행 전에 호출됨"""
        # 테스트용 로거 설정
        self.logger = BaseLogger("TestLogger", log_dir=os.path.join(project_root, 'tests', 'logs'))
        
    def tearDown(self):
        """각 테스트 메서드 실행 후에 호출됨"""
        # 로거 핸들러 정리
        if self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
        
        # ModelEvaluator 인스턴스 생성
        self.evaluator = ModelEvaluator(logger=self.logger)
        
        # 테스트용 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        # 실제 라벨 데이터 - 3개 클래스(0, 1, 2)
        self.y_true = np.random.randint(0, 3, n_samples)
        
        # 예측된 라벨 데이터 - 정확도를 조절하여 생성
        # 약 70% 정확도를 갖도록 설정
        self.y_pred = np.copy(self.y_true)
        mask = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
        self.y_pred[mask] = np.random.randint(0, 3, mask.sum())
        
        # 예측 확률 - 3개 클래스에 대한 확률 값
        # 예측 클래스에 높은 확률을 부여하도록 설정
        self.y_prob = np.zeros((n_samples, 3))
        for i in range(n_samples):
            probs = np.random.uniform(0, 0.3, 3)
            probs[self.y_pred[i]] = np.random.uniform(0.5, 0.9)
            probs = probs / probs.sum()  # 확률의 합이 1이 되도록 정규화
            self.y_prob[i] = probs
        
        # 테스트 디렉토리 설정
        self.test_dir = os.path.join(project_root, 'tests', 'results')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 결과 저장 경로
        self.result_path = os.path.join(self.test_dir, 'test_eval_results')
        os.makedirs(self.result_path, exist_ok=True)
        
        # 모델 이름
        self.model_name = "test_model"
    
    def _deprecated_test_evaluate_classifier(self):
        """evaluate_classifier 메서드 테스트"""
        # 분류기 평가 실행
        metrics = self.evaluator.evaluate_classifier(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
            model_name=self.model_name,
            result_path=self.result_path
        )
        
        # 메트릭 결과 확인
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # 메트릭 값 범위 확인 (0~1 사이의 값)
        for metric_name, value in metrics.items():
            if metric_name != 'model_name':  # model_name은 문자열이므로 제외
                self.assertTrue(0 <= value <= 1, f"{metric_name} 값이 0~1 범위를 벗어남: {value}")
        
        # 모델 이름이 올바르게 저장되었는지 확인
        self.assertEqual(metrics['model_name'], self.model_name)
    
    def _deprecated_test_confusion_matrix(self):
        """plot_confusion_matrix 메서드 테스트"""
        # 혼동 행렬 생성 및 저장
        cm_path = self.evaluator.plot_confusion_matrix(
            y_true=self.y_true,
            y_pred=self.y_pred,
            class_names=['class_0', 'class_1', 'class_2'],
            model_name=self.model_name,
            result_path=self.result_path
        )
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(cm_path))
        
        # 올바른 경로에 저장되었는지 확인
        expected_filename = f"{self.model_name}_confusion_matrix.png"
        self.assertTrue(cm_path.endswith(expected_filename))
    
    def _deprecated_test_feature_importance_plot(self):
        """plot_feature_importance 메서드 테스트"""
        # 특성 중요도 데이터 생성
        feature_names = [f"feature_{i}" for i in range(10)]
        importances = np.random.uniform(0, 1, 10)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 중요도 순으로 정렬
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 특성 중요도 플롯 생성 및 저장
        fi_path = self.evaluator.plot_feature_importance(
            importance_df=importance_df,
            top_n=5,
            model_name=self.model_name,
            result_path=self.result_path
        )
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(fi_path))
        
        # 올바른 경로에 저장되었는지 확인
        expected_filename = f"{self.model_name}_feature_importance.png"
        self.assertTrue(fi_path.endswith(expected_filename))
    
    def _deprecated_test_cross_validation(self):
        """cross_validate 메서드 테스트"""
        # 테스트용 모델 클래스 (간단한 더미 모델)
        class DummyModel:
            def __init__(self):
                self.fitted = False
                self.X = None
                self.y = None
            
            def train(self, X, y):
                self.fitted = True
                self.X = X
                self.y = y
                return self
            
            def predict(self, X):
                # 무작위 예측
                return np.random.randint(0, 3, len(X))
            
            def predict_proba(self, X):
                # 무작위 확률 예측
                probs = np.random.uniform(0, 1, (len(X), 3))
                return probs / probs.sum(axis=1, keepdims=True)
        
        # 피처 데이터
        X = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, len(self.y_true)) for i in range(10)
        })
        
        # 타겟 데이터
        y = pd.Series(self.y_true)
        
        # 교차 검증 실행
        cv_results = self.evaluator.cross_validate(
            model_class=DummyModel,
            X=X,
            y=y,
            n_splits=2,  # 빠른 테스트를 위한 작은 값
            model_params={},
            cv_params={},
            model_name=self.model_name,
            result_path=self.result_path
        )
        
        # 결과 확인
        self.assertIsInstance(cv_results, dict)
        self.assertIn('test_accuracy', cv_results)
        self.assertIn('test_precision', cv_results)
        self.assertIn('test_recall', cv_results)
        self.assertIn('test_f1', cv_results)
        
        # 각 메트릭 결과가 리스트인지 확인
        for metric_name, values in cv_results.items():
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), 2)  # n_splits=2를 사용했으므로


class TestModelEvaluatorV2(unittest.TestCase):
    """ModelEvaluator 클래스의 실제 구현에 맞춘 신규 테스트"""

    def setUp(self):
        """각 테스트 메서드 실행 전에 호출됨"""
        # 테스트용 로거 및 평가자
        self.logger = BaseLogger("TestLoggerV2", log_dir=os.path.join(project_root, 'tests', 'logs'))
        self.evaluator = ModelEvaluator(logger=self.logger)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 100
        self.y_true = np.random.randint(0, 3, n_samples)
        self.y_pred = np.random.randint(0, 3, n_samples)
        
        # 결과 저장 디렉토리 생성
        self.result_path = os.path.join(project_root, 'tests', 'results_v2')
        os.makedirs(self.result_path, exist_ok=True)
        
    def tearDown(self):
        """각 테스트 메서드 실행 후에 호출됨"""
        # 로거 핸들러 정리
        if hasattr(self, 'logger') and self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
                
        # 결과 파일 정리
        if hasattr(self, 'result_path') and os.path.exists(self.result_path):
            for file in os.listdir(self.result_path):
                try:
                    os.remove(os.path.join(self.result_path, file))
                except PermissionError:
                    pass  # Windows에서 파일이 사용 중일 때 발생할 수 있음
            try:
                os.rmdir(self.result_path)
            except (PermissionError, OSError):
                pass  # 디렉토리가 사용 중이거나 비어있지 않은 경우

    def test_evaluate(self):
        class DummyModel:
            def predict(self, X):
                np.random.seed(0)
                return np.random.randint(0, 3, len(X))
        X = pd.DataFrame({f'feature_{i}': np.random.randn(100) for i in range(5)})
        y = pd.Series(self.y_true)
        model = DummyModel()
        metrics = self.evaluator.evaluate(model, X, y)
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_plot_confusion_matrix_actual(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        save_path = os.path.join(self.result_path, 'confusion_matrix.png')
        self.evaluator.plot_confusion_matrix(cm=cm, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_feature_importance_actual(self):
        importance_df = pd.DataFrame({
            'feature': [f'f{i}' for i in range(10)],
            'importance': np.random.rand(10)
        }).sort_values('importance', ascending=False)
        save_path = os.path.join(self.result_path, 'feature_importance.png')
        self.evaluator.plot_feature_importance(importance_df=importance_df, top_n=5, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_cross_validate_actual(self):
        class DummyModel:
            def train(self, X, y, *args, **kwargs):
                return self
            def predict(self, X):
                return np.random.randint(0, 3, len(X))
        model = DummyModel()
        X = pd.DataFrame({f'feature_{i}': np.random.randn(100) for i in range(5)})
        y = pd.Series(self.y_true)
        cv_results = self.evaluator.cross_validate(model=model, X=X, y=y, n_splits=2, random_state=42)
        self.assertIn('fold_metrics', cv_results)
        self.assertIn('overall_metrics', cv_results)

if __name__ == '__main__':
    unittest.main()
