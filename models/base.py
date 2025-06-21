#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기본 모델 클래스 정의.
모델 학습 및 평가에 필요한 기본 클래스들을 제공합니다.
"""
import os
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseLogger:
    """로깅 기능을 제공하는 기본 클래스입니다."""
    
    def __init__(self, name: str, log_dir: str = None):
        """
        로거 초기화.
        
        Args:
            name: 로거 이름
            log_dir: 로그 파일 저장 디렉토리. None이면 기본 위치 사용.
        """
        self.name = name
        
        # 로그 디렉토리 설정
        if log_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(project_root, 'logs')
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 파일 경로
        log_filename = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file = os.path.join(log_dir, log_filename)
        
        # 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"로깅 설정 완료. 로그 파일: {self.log_file}")
    
    def info(self, message: str) -> None:
        """정보 로그 메시지를 기록합니다."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """경고 로그 메시지를 기록합니다."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """오류 로그 메시지를 기록합니다."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """디버그 로그 메시지를 기록합니다."""
        self.logger.debug(message)


class BaseModel(ABC):
    """모델의 기본 인터페이스를 정의하는 추상 클래스입니다."""
    
    def __init__(self, params: Dict[str, Any] = None, logger: BaseLogger = None):
        """
        모델 초기화.
        
        Args:
            params: 모델 파라미터
            logger: 로거 인스턴스
        """
        self.params = params or {}
        self.logger = logger or BaseLogger(self.__class__.__name__)
        self.model = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            **kwargs: 추가 파라미터
            
        Returns:
            학습된 모델
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측을 수행합니다.
        
        Args:
            X: 피처 데이터
            
        Returns:
            예측 결과
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        모델을 파일로 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        저장된 모델을 로드합니다.
        
        Args:
            filepath: 모델 파일 경로
            
        Returns:
            로드된 모델 인스턴스
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        특성 중요도를 반환합니다.
        
        Returns:
            특성 중요도가 포함된 데이터프레임
        """
        pass


class BaseDataProcessor(ABC):
    """데이터 처리 기능을 제공하는 기본 클래스입니다."""
    
    def __init__(self, logger: BaseLogger = None):
        """
        초기화.
        
        Args:
            logger: 로거 인스턴스
        """
        self.logger = logger or BaseLogger(self.__class__.__name__)
    
    @abstractmethod
    def load(self, filepath: str) -> pd.DataFrame:
        """
        데이터 파일을 로드합니다.
        
        Args:
            filepath: 데이터 파일 경로
            
        Returns:
            로드된 데이터프레임
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터를 전처리합니다.
        
        Args:
            data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        pass
    
    @abstractmethod
    def split_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        피처와 타겟을 분리합니다.
        
        Args:
            data: 전체 데이터
            
        Returns:
            (피처, 타겟) 튜플
        """
        pass


class BaseTrainer(ABC):
    """모델 학습 및 평가를 관리하는 기본 클래스입니다."""
    
    def __init__(self, model: BaseModel, logger: BaseLogger = None):
        """
        초기화.
        
        Args:
            model: 학습할 모델 인스턴스
            logger: 로거 인스턴스
        """
        self.model = model
        self.logger = logger or BaseLogger(self.__class__.__name__)
        self.metrics = {}
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            **kwargs: 추가 학습 파라미터
            
        Returns:
            학습된 모델
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        모델을 평가합니다.
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            
        Returns:
            평가 지표
        """
        pass
    
    @abstractmethod
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int) -> Dict[str, List[float]]:
        """
        교차 검증을 수행합니다.
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            n_splits: 폴드 수
            
        Returns:
            폴드별 평가 지표
        """
        pass


class ConfigManager:
    """설정 관리 클래스입니다."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        초기화.
        
        Args:
            config_dict: 설정 딕셔너리
        """
        self.config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값을 가져옵니다.
        
        Args:
            key: 설정 키
            default: 키가 없을 때 반환할 기본값
            
        Returns:
            설정값
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        설정값을 설정합니다.
        
        Args:
            key: 설정 키
            value: 설정값
        """
        self.config[key] = value
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        여러 설정을 한꺼번에 업데이트합니다.
        
        Args:
            new_config: 새로운 설정 딕셔너리
        """
        self.config.update(new_config)
    
    def save(self, filepath: str) -> None:
        """
        설정을 파일로 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ConfigManager':
        """
        파일에서 설정을 로드합니다.
        
        Args:
            filepath: 설정 파일 경로
            
        Returns:
            설정 관리자 인스턴스
        """
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return cls(config)
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            return cls()
