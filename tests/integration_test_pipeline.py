import os
import sys
import unittest
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.base import BaseLogger
from models.data_loader import DataLoader
from models.training import XGBoostModel
from models.evaluation import ModelEvaluator


class TestPipelineIntegration(unittest.TestCase):
    """DataLoader -> XGBoostModel -> ModelEvaluator 통합 파이프라인 테스트"""

    def setUp(self):
        # 로거 설정 (테스트 로그 디렉토리 사용)
        self.logger = BaseLogger("IntegrationLogger", log_dir=os.path.join(project_root, 'tests', 'logs'))

        # 간단한 샘플 데이터 생성
        self.raw_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103, 104],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'feature2': [1.1, 1.0, 1.2, 1.1, 1.3, 1.2],
            'feature3': [2.1, np.nan, 2.3, 2.4, 2.5, 2.6],
            'bin': [0, 1, 2, 0, 1, 2]
        })

        # DataLoader 인스턴스 (언더샘플링 비활성화)
        self.data_loader = DataLoader(drop_columns=[], target_column='bin', handle_imbalance=False, logger=self.logger)

        # 전처리 및 특성/타겟 분리
        processed = self.data_loader.preprocess(self.raw_data)
        self.X, self.y = self.data_loader.split_features_target(processed)

        # XGBoost 모델 (작은 파라미터로 빠르게 학습)
        self.model = XGBoostModel(params={
            'n_estimators': 10,
            'max_depth': 2,
            'learning_rate': 0.3,
            'num_class': 3,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }, logger=self.logger)

    def tearDown(self):
        # 로거 핸들러 정리
        if hasattr(self, 'logger') and self.logger and getattr(self.logger, 'logger', None):
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)

    def test_full_pipeline(self):
        """전체 파이프라인 학습 및 평가가 정상 동작하는지 테스트"""
        # 모델 학습
        self.model.train(self.X, self.y, verbose=False)

        # 평가
        evaluator = ModelEvaluator(logger=self.logger)
        metrics = evaluator.evaluate(self.model, self.X, self.y)

        # 주요 지표가 존재하는지 확인
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)


if __name__ == '__main__':
    unittest.main()
