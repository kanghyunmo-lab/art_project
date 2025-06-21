#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 평가 및 성능 측정 기능을 제공합니다.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.base import BaseLogger, BaseModel, BaseTrainer


class ModelEvaluator:
    """모델 평가 및 성능 측정을 담당하는 클래스입니다."""
    
    def __init__(
        self, 
        class_names: Dict[int, str] = None,
        logger: BaseLogger = None
    ):
        """
        초기화.
        
        Args:
            class_names: 클래스 이름 매핑 딕셔너리
            logger: 로거 인스턴스
        """
        self.logger = logger or BaseLogger("ModelEvaluator")
        self.class_names = class_names or {0: '매도', 1: '중립', 2: '매수'}
    
    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        모델을 평가합니다.
        
        Args:
            model: 평가할 모델
            X: 피처 데이터
            y: 실제 레이블
            
        Returns:
            평가 지표 딕셔너리
        """
        try:
            # 예측
            y_pred = model.predict(X)
            
            # 주요 지표 계산
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision_macro': precision_score(y, y_pred, average='macro'),
                'recall_macro': recall_score(y, y_pred, average='macro'),
                'f1_macro': f1_score(y, y_pred, average='macro'),
                'precision_weighted': precision_score(y, y_pred, average='weighted'),
                'recall_weighted': recall_score(y, y_pred, average='weighted'),
                'f1_weighted': f1_score(y, y_pred, average='weighted')
            }
            
            # 클래스별 지표 계산
            class_report = classification_report(y, y_pred, output_dict=True)
            
            # 클래스별 지표 로깅 및 딕셔너리에 추가
            for cls in sorted(self.class_names.keys()):
                cls_str = str(cls)
                if cls_str in class_report:
                    cls_name = self.class_names[cls]
                    cls_metrics = class_report[cls_str]
                    
                    metrics[f'{cls_name}_precision'] = cls_metrics['precision']
                    metrics[f'{cls_name}_recall'] = cls_metrics['recall']
                    metrics[f'{cls_name}_f1'] = cls_metrics['f1-score']
                    
                    self.logger.info(f"\n{cls_name} ({cls}) 클래스 성능:")
                    self.logger.info(f"  - 정밀도: {cls_metrics['precision']:.4f}")
                    self.logger.info(f"  - 재현율: {cls_metrics['recall']:.4f}")
                    self.logger.info(f"  - F1 점수: {cls_metrics['f1-score']:.4f}")
            
            # 혼동 행렬 계산
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # 주요 지표 로깅
            self.logger.info("\n" + "="*50)
            self.logger.info("모델 평가 결과:")
            self.logger.info(f"정확도: {metrics['accuracy']:.4f}")
            self.logger.info(f"매크로 평균 - 정밀도: {metrics['precision_macro']:.4f}, 재현율: {metrics['recall_macro']:.4f}, F1: {metrics['f1_macro']:.4f}")
            self.logger.info(f"가중 평균 - 정밀도: {metrics['precision_weighted']:.4f}, 재현율: {metrics['recall_weighted']:.4f}, F1: {metrics['f1_weighted']:.4f}")
            self.logger.info("="*50 + "\n")
            
            return metrics
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {e}")
            raise
    
    def cross_validate(
        self, 
        model: BaseModel, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        교차 검증을 수행합니다.
        
        Args:
            model: 평가할 모델
            X: 피처 데이터
            y: 실제 레이블
            n_splits: 폴드 수
            random_state: 랜덤 시드
            
        Returns:
            폴드별 평가 지표 딕셔너리
        """
        self.logger.info(f"{n_splits}겹 교차 검증 시작...")
        
        # Stratified K-Fold 설정
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # 폴드별 지표 저장을 위한 딕셔너리
        fold_metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_weighted': [],
            'recall_weighted': [],
            'f1_weighted': []
        }
        
        # 클래스별 지표 저장
        class_metrics = {}
        for cls in self.class_names.keys():
            cls_name = self.class_names[cls]
            class_metrics[f'{cls_name}_precision'] = []
            class_metrics[f'{cls_name}_recall'] = []
            class_metrics[f'{cls_name}_f1'] = []
        
        # 교차 검증 수행
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            self.logger.info(f"\n폴드 {fold_idx}/{n_splits} 평가 시작")
            
            # 데이터 분할
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # 클래스 가중치 계산
                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=y_train_fold
                )
                
                # 모델 학습
                model.train(
                    X_train_fold, 
                    y_train_fold, 
                    sample_weight=sample_weights,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=10,
                    verbose=10
                )
                
                # 검증 세트 평가
                fold_result = self.evaluate(model, X_val_fold, y_val_fold)
                
                # 지표 저장
                for metric in fold_metrics.keys():
                    fold_metrics[metric].append(fold_result[metric])
                
                # 클래스별 지표 저장
                for metric_name in class_metrics.keys():
                    if metric_name in fold_result:
                        class_metrics[metric_name].append(fold_result[metric_name])
                
            except Exception as e:
                self.logger.error(f"폴드 {fold_idx} 평가 중 오류 발생: {e}")
                # 오류가 발생해도 계속 진행
                continue
        
        # 전체 평가 지표 계산
        overall_metrics = {}
        
        # 일반 지표
        for metric_name, values in fold_metrics.items():
            if values:  # 빈 리스트가 아닌 경우에만 처리
                overall_metrics[f'{metric_name}_mean'] = np.mean(values)
                overall_metrics[f'{metric_name}_std'] = np.std(values)
                
                self.logger.info(f"{metric_name}: {overall_metrics[f'{metric_name}_mean']:.4f} ± {overall_metrics[f'{metric_name}_std']:.4f}")
        
        # 클래스별 지표
        for metric_name, values in class_metrics.items():
            if values:  # 빈 리스트가 아닌 경우에만 처리
                overall_metrics[f'{metric_name}_mean'] = np.mean(values)
                overall_metrics[f'{metric_name}_std'] = np.std(values)
        
        # 폴드별 지표와 최종 지표 모두 반환
        result = {
            'fold_metrics': fold_metrics,
            'class_metrics': class_metrics,
            'overall_metrics': overall_metrics
        }
        
        return result
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        figsize: Tuple[int, int] = (10, 8),
        save_path: str = None
    ) -> None:
        """
        혼동 행렬을 시각화합니다.
        
        Args:
            cm: 혼동 행렬
            figsize: 그림 크기
            save_path: 저장할 파일 경로 (None이면 저장하지 않음)
        """
        try:
            class_names = list(self.class_names.values())
            
            plt.figure(figsize=figsize)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('혼동 행렬')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # 값 표시
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('실제 레이블')
            plt.xlabel('예측 레이블')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"혼동 행렬 이미지 저장 완료: {save_path}")
            
            plt.close()
        except Exception as e:
            self.logger.error(f"혼동 행렬 시각화 중 오류 발생: {e}")
            plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None
    ) -> None:
        """
        특성 중요도를 시각화합니다.
        
        Args:
            importance_df: 특성 중요도 데이터프레임 (feature, importance 컬럼 포함)
            top_n: 표시할 상위 특성 수
            figsize: 그림 크기
            save_path: 저장할 파일 경로 (None이면 저장하지 않음)
        """
        try:
            # 상위 N개 특성 추출
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=figsize)
            plt.barh(
                np.arange(len(top_features)),
                top_features['importance'].values,
                align='center'
            )
            plt.yticks(np.arange(len(top_features)), top_features['feature'].values)
            plt.title(f'상위 {top_n}개 특성 중요도')
            plt.xlabel('중요도')
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"특성 중요도 이미지 저장 완료: {save_path}")
            
            plt.close()
        except Exception as e:
            self.logger.error(f"특성 중요도 시각화 중 오류 발생: {e}")
            plt.close()
