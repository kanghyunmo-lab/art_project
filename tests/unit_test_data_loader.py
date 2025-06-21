#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models.data_loader 모듈의 테스트 코드
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 테스트할 모듈 가져오기
from models.data_loader import DataLoader
from models.base import BaseLogger


class TestDataLoader(unittest.TestCase):
    """DataLoader 클래스의 단위 테스트를 위한 테스트 케이스"""

    def setUp(self):
        """각 테스트 메서드 실행 전에 호출됨"""
        # 테스트용 로거 (출력 없음)
        self.logger = BaseLogger("TestLogger", log_dir=os.path.join(project_root, 'tests', 'logs'))
        
        # 테스트 디렉토리 생성
        self.test_dir = os.path.join(project_root, 'tests', 'data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 테스트용 DataLoader 인스턴스 생성
        self.data_loader = DataLoader(
            drop_columns=['volume', 'ret', 'trgt'],
            target_column='bin',
            handle_imbalance=False,
            logger=self.logger
        )
        
        # 테스트 데이터 생성
        self.test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0],
            'close': [101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300],
            'ret': [0.01, 0.02, 0.03, 0.04],
            'trgt': [0.02, 0.03, 0.04, 0.05],
            'bin': [1, 0, 1, 1],
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [1.1, 1.2, 1.3, 1.4],
            'feature3': [2.1, 2.2, np.nan, 2.4]  # NaN 테스트용
        })
        
        # 테스트 파일 생성
        self.test_file = os.path.join(self.test_dir, 'test_data.parquet')
        self.test_data.to_parquet(self.test_file, index=False)
        
    def tearDown(self):
        """각 테스트 메서드 실행 후에 호출됨"""
        # 로거 핸들러 정리
        if hasattr(self, 'logger') and self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
                
        # 테스트 파일 정리
        if hasattr(self, 'test_file') and os.path.exists(self.test_file):
            try:
                os.remove(self.test_file)
            except PermissionError:
                pass  # Windows에서 파일이 사용 중일 때 발생할 수 있음
    
    def test_load(self):
        """load 메서드 테스트"""
        # 파일 로드 테스트
        loaded_data = self.data_loader.load(self.test_file)
        
        # 로드된 데이터 검증
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), len(self.test_data))
        self.assertEqual(set(loaded_data.columns), set(self.test_data.columns))
        
        # 존재하지 않는 파일 로드 시 예외 발생 테스트
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load(os.path.join(self.test_dir, 'non_existent_file.parquet'))
    
    def test_preprocess(self):
        """preprocess 메서드 테스트"""
        processed_data = self.data_loader.preprocess(self.test_data.copy())
        
        # 전처리 후 검증
        # 1. NaN 값이 모두 처리되었는지 확인
        self.assertFalse(processed_data['feature3'].isna().any())
        
        # 2. 타겟 컬럼이 존재하는지 확인
        self.assertIn(self.data_loader.target_column, processed_data.columns)
        
        # datetime 컬럼 제거 확인
        self.assertNotIn('t1', processed_data.columns)
        
        # 빈 데이터프레임 전처리 시 예외 발생 테스트
        with self.assertRaises(ValueError):
            self.data_loader.preprocess(pd.DataFrame())
    
    def test_split_features_target(self):
        """split_features_target 메서드 테스트"""
        # 전처리된 데이터로 분할 테스트
        processed_data = self.data_loader.preprocess(self.test_data.copy())
        X, y = self.data_loader.split_features_target(processed_data)
        
        # 1. 특성과 타겟의 크기가 일치하는지 확인
        self.assertEqual(len(y), len(processed_data))
        
        # 2. 타겟 컬럼이 특성에서 제외되었는지 확인
        self.assertNotIn(self.data_loader.target_column, X.columns)
        
        # 3. 타겟이 올바른 형식인지 확인
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual(y.name, self.data_loader.target_column)
        
        # 드롭 컬럼 제외 확인
        for col in self.data_loader.drop_columns:
            if col in self.test_data.columns:
                self.assertNotIn(col, X.columns)
        
        # 타겟 시리즈 확인
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(y), len(self.test_data))
        
        # 클래스 매핑 확인 (모든 클래스가 0, 1, 2 범위 내에 있어야 함)
        self.assertTrue(set(y).issubset({0, 1, 2}))
        
        # 타겟 컬럼이 없는 경우 예외 발생 테스트
        with self.assertRaises(ValueError):
            self.data_loader.split_features_target(self.test_data.drop(columns=['bin']))


if __name__ == '__main__':
    unittest.main()
