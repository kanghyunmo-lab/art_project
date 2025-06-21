#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 학습 및 평가 기능을 제공합니다.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.base import BaseLogger, BaseModel, BaseTrainer


class XGBoostModel(BaseModel):
    """XGBoost 기반 분류 모델입니다."""
    
    def __init__(self, params: Dict[str, Any] = None, logger: BaseLogger = None):
        """
        초기화.
        
        Args:
            params: XGBoost 모델 파라미터
            logger: 로거 인스턴스
        """
        super().__init__(params=params, logger=logger or BaseLogger("XGBoostModel"))
        
        # 기본 모델 파라미터
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'num_class': 3,  # 3개 클래스 분류
            'verbosity': 1  # 로그 레벨 설정
        }
        
        # 사용자 파라미터로 기본 파라미터 업데이트
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> xgb.XGBClassifier:
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            **kwargs: 추가 학습 파라미터
                - sample_weight: 샘플 가중치
                - eval_set: 검증 데이터셋
                - early_stopping_rounds: 조기 종료 라운드 수
                - verbose: 학습 과정 출력 여부
        
        Returns:
            학습된 모델
        """
        self.logger.info("XGBoost 모델 학습 시작...")
        self.feature_names = X.columns.tolist()
        
        # 추가 학습 파라미터 처리
        fit_params = {
            'verbose': kwargs.get('verbose', 10)
        }
        
        # 검증 세트가 제공되면 추가
        eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        
        # 샘플 가중치가 제공되면 추가
        sample_weight = kwargs.get('sample_weight')
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # 모델 초기화 및 학습
        try:
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X, y, **fit_params)
            self.logger.info("모델 학습 완료")
            return self.model
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측을 수행합니다.
        
        Args:
            X: 피처 데이터
            
        Returns:
            예측 레이블
        """
        if self.model is None:
            self.logger.error("모델이 학습되지 않았습니다.")
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        클래스별 확률을 예측합니다.
        
        Args:
            X: 피처 데이터
            
        Returns:
            클래스별 확률
        """
        if self.model is None:
            self.logger.error("모델이 학습되지 않았습니다.")
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """
        모델을 파일로 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        if self.model is None:
            self.logger.error("저장할 모델이 없습니다.")
            raise ValueError("저장할 모델이 없습니다.")
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            self.logger.info(f"모델이 성공적으로 저장되었습니다: {filepath}")
            
            # 특성 이름도 저장
            feature_names_path = os.path.join(
                os.path.dirname(filepath),
                f"{os.path.splitext(os.path.basename(filepath))[0]}_features.txt"
            )
            with open(feature_names_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.feature_names))
            self.logger.info(f"특성 이름이 저장되었습니다: {feature_names_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostModel':
        """
        저장된 모델을 로드합니다.
        
        Args:
            filepath: 모델 파일 경로
            
        Returns:
            로드된 모델 인스턴스
        """
        instance = cls()
        try:
            instance.model = joblib.load(filepath)
            
            # 특성 이름 로드 시도
            feature_names_path = os.path.join(
                os.path.dirname(filepath),
                f"{os.path.splitext(os.path.basename(filepath))[0]}_features.txt"
            )
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    instance.feature_names = f.read().splitlines()
            
            instance.logger.info(f"모델을 성공적으로 로드했습니다: {filepath}")
            return instance
        except Exception as e:
            instance.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        특성 중요도를 반환합니다.
        
        Returns:
            특성 중요도가 포함된 데이터프레임
        """
        if self.model is None:
            self.logger.error("모델이 학습되지 않았습니다.")
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특성 이름이 없으면 기본 이름 생성
        if self.feature_names is None or len(self.feature_names) != len(self.model.feature_importances_):
            self.feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        # 특성 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
