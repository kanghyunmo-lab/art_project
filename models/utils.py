#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 학습 및 평가를 위한 유틸리티 함수를 제공합니다.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.base import BaseLogger


def save_json(data: Dict, filepath: str) -> None:
    """
    딕셔너리를 JSON 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터 딕셔너리
        filepath: 저장할 파일 경로
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")
        return False


def load_json(filepath: str) -> Dict:
    """
    JSON 파일을 로드합니다.
    
    Args:
        filepath: 로드할 파일 경로
        
    Returns:
        로드된 딕셔너리
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 파일 로드 중 오류 발생: {e}")
        return {}


def get_timestamp() -> str:
    """
    현재 시간 문자열을 반환합니다.
    
    Returns:
        YYYYMMDD_HHMMSS 형식의 시간 문자열
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    실험 결과를 저장할 디렉토리를 생성합니다.
    
    Args:
        base_dir: 기본 디렉토리 경로
        experiment_name: 실험 이름 (None이면 시간 기반으로 생성)
        
    Returns:
        생성된 디렉토리 경로
    """
    if experiment_name is None:
        experiment_name = f"experiment_{get_timestamp()}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def format_time(seconds: float) -> str:
    """
    초 단위 시간을 읽기 쉬운 형식으로 변환합니다.
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        형식화된 시간 문자열
    """
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}분 {int(seconds)}초"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"


def convert_to_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환합니다.
    
    Args:
        obj: 변환할 객체
        
    Returns:
        직렬화 가능한 객체
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, set):
        return list(convert_to_serializable(v) for v in obj)
    else:
        # 기본 타입은 그대로 반환
        return obj
