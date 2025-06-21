#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 로드 및 전처리 기능을 제공합니다.
학습 데이터 로드, 전처리, 특성/타겟 분리 등의 기능을 담당합니다.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.base import BaseLogger, BaseDataProcessor


class DataLoader(BaseDataProcessor):
    """데이터 로드 및 전처리를 담당하는 클래스입니다."""
    
    def __init__(
        self,
        drop_columns: List[str] = None,
        target_column: str = 'bin',
        class_mapping: Dict[int, int] = None,
        handle_imbalance: bool = True,
        min_samples_per_class: int = 100,
        logger: BaseLogger = None
    ):
        """
        초기화.
        
        Args:
            drop_columns: 제외할 컬럼 목록
            target_column: 타겟 컬럼 이름
            class_mapping: 클래스 값 매핑 딕셔너리
            handle_imbalance: 클래스 불균형 처리 여부
            min_samples_per_class: 클래스당 최소 샘플 수
            logger: 로거 인스턴스
        """
        super().__init__(logger=logger or BaseLogger("DataLoader"))
        self.drop_columns = drop_columns or ['open', 'high', 'low', 'close', 'volume', 'ret', 'trgt', 'bin', 't1']
        self.target_column = target_column
        self.class_mapping = class_mapping or {-1: 0, 0: 1, 1: 2}  # 매도(-1) -> 0, 중립(0) -> 1, 매수(1) -> 2
        self.handle_imbalance = handle_imbalance
        self.min_samples_per_class = min_samples_per_class
        self.class_names = {0: '매도', 1: '중립', 2: '매수'}
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Parquet 파일에서 데이터를 로드합니다.
        
        Args:
            filepath: 데이터 파일 경로
            
        Returns:
            로드된 데이터프레임
        
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            Exception: 데이터 로드 중 오류 발생
        """
        if not os.path.exists(filepath):
            self.logger.error(f"데이터 파일을 찾을 수 없습니다: {filepath}")
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {filepath}")
        
        self.logger.info(f"데이터 로드 중: {filepath}")
        try:
            df = pd.read_parquet(filepath)
            self.logger.info(f"데이터 로드 완료. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터를 전처리합니다.
        - NaN 값 처리
        - 불필요한 컬럼 제거
        
        Args:
            data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        if data.empty:
            self.logger.error("입력 데이터프레임이 비어 있습니다.")
            raise ValueError("입력 데이터프레임이 비어 있습니다.")
        
        self.logger.info("데이터 전처리 시작...")
        
        # 복사본 생성
        df = data.copy()
        
        # datetime 타입 컬럼 제거
        df = df.select_dtypes(exclude=['datetime64[ns, UTC]', 'datetime64[ns]'])
        
        # NaN 값 처리
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.logger.warning(f"컬럼 '{col}'에 {null_count}개 NaN 값이 존재합니다.")
                # 숫자형 컬럼은 0으로, 범주형 컬럼은 최빈값으로 대체
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '', inplace=True)
        
        self.logger.info(f"전처리 완료. Shape: {df.shape}")
        return df
    
    def split_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        피처와 타겟을 분리합니다.
        
        Args:
            data: 전체 데이터
            
        Returns:
            (피처 데이터프레임, 타겟 시리즈) 튜플
        
        Raises:
            ValueError: 타겟 컬럼이 없거나 데이터프레임이 비어있을 경우
        """
        if data is None or data.empty:
            self.logger.error("입력 데이터프레임이 비어 있습니다.")
            raise ValueError("입력 데이터프레임이 비어 있습니다.")
        
        if self.target_column not in data.columns:
            self.logger.error(f"타겟 컬럼 '{self.target_column}'을(를) 찾을 수 없습니다.")
            raise ValueError(f"타겟 컬럼 '{self.target_column}'을(를) 찾을 수 없습니다.")
        
        # 타겟 추출
        y = data[self.target_column].copy()
        
        # NaN 값 확인 및 처리
        if y.isnull().any():
            null_count = y.isnull().sum()
            self.logger.warning(f"타겟 변수에 {null_count}개의 NaN 값이 있습니다. 중립(0)으로 채웁니다.")
            y = y.fillna(0)
        
        # 클래스 매핑 적용
        y = y.map(self.class_mapping).astype(int)
        
        # 피처 컬럼 추출 (타겟 컬럼 및 제외할 컬럼 제외)
        drop_cols = [col for col in self.drop_columns if col in data.columns and col != self.target_column]
        X = data.drop(columns=[self.target_column] + drop_cols, errors='ignore')
        
        # 클래스 분포 출력
        self._log_class_distribution(y)
        
        # 클래스 불균형 처리
        if self.handle_imbalance:
            X, y = self._handle_class_imbalance(X, y)
        
        self.logger.info(f"X 크기: {X.shape}, y 크기: {len(y)}")
        return X, y
    
    def _log_class_distribution(self, y: pd.Series) -> None:
        """
        클래스 분포 정보를 로깅합니다.
        
        Args:
            y: 타겟 시리즈
        """
        class_dist = y.value_counts().sort_index()
        total_samples = len(y)
        
        self.logger.info("\n" + "="*50)
        self.logger.info("클래스 분포:")
        for cls, count in class_dist.items():
            cls_name = self.class_names.get(cls, str(cls))
            self.logger.info(f"{cls_name} ({cls}): {count}개 ({count/total_samples*100:.2f}%)")
        self.logger.info("="*50 + "\n")
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        클래스 불균형을 처리합니다(언더샘플링).
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            
        Returns:
            불균형이 처리된 (X, y) 튜플
        """
        class_dist = y.value_counts().sort_index()
        min_samples = class_dist.min()
        
        # 최소 샘플 수가 지정된 값보다 작으면 언더샘플링
        if min_samples < self.min_samples_per_class:
            self.logger.warning(f"클래스 불균형이 심각합니다. 언더샘플링을 수행합니다.")
            
            # 클래스별 인덱스 추출
            idx_list = []
            for cls in class_dist.index:
                cls_indices = y[y == cls].index
                if len(cls_indices) > min_samples:
                    # 다수 클래스는 무작위로 min_samples 개수만큼 샘플링
                    sampled_indices = np.random.choice(
                        cls_indices, 
                        size=min_samples, 
                        replace=False
                    )
                    idx_list.extend(sampled_indices)
                else:
                    # 소수 클래스는 모두 사용
                    idx_list.extend(cls_indices)
            
            # 샘플링된 데이터 추출
            X_balanced = X.loc[idx_list].copy()
            y_balanced = y.loc[idx_list].copy()
            
            self.logger.info(f"언더샘플링 후 데이터 크기: {len(X_balanced)}")
            self.logger.info("새로운 클래스 분포:")
            for cls, count in y_balanced.value_counts().sort_index().items():
                cls_name = self.class_names.get(cls, str(cls))
                self.logger.info(f"{cls_name} ({cls}): {count}개 ({count/len(y_balanced)*100:.2f}%)")
                
            return X_balanced, y_balanced
        else:
            return X, y
