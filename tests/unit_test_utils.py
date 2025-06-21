#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models.utils 모듈의 테스트 코드
"""
import os
import sys
import json
import unittest
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 테스트할 모듈 가져오기
from models.base import BaseLogger
from models.utils import (
    save_json, load_json, get_timestamp, 
    create_experiment_dir, format_time, convert_to_serializable
)


class TestUtils(unittest.TestCase):
    """유틸리티 함수들에 대한 테스트 케이스"""

    def setUp(self):
        """각 테스트 메서드 실행 전에 호출됨"""
        # 테스트 디렉토리 생성
        self.test_dir = os.path.join(project_root, 'tests', 'utils_test')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 테스트용 로거 설정
        self.logger = BaseLogger("TestUtilsLogger", log_dir=os.path.join(project_root, 'tests', 'logs'))
        
        # 테스트용 데이터
        self.test_data = {
            'string': 'test',
            'number': 42,
            'boolean': True,
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'nested': {'c': [1, 2, {'d': 3}]}
        }

    def tearDown(self):
        """각 테스트 메서드 실행 후에 호출됨"""
        # 로거 핸들러 정리
        if hasattr(self, 'logger') and self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
        # 로거 핸들러 정리
        if self.logger and self.logger.logger:
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)
        
        # 테스트 파일 정리
        import shutil
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except (PermissionError, OSError):
                pass  # Windows에서는 파일이 사용 중이면 삭제할 수 없음
    
    def test_save_load_json(self):
        """save_json 및 load_json 함수 테스트"""
        # JSON 파일 경로
        json_path = os.path.join(self.test_dir, 'test_data.json')
        
        # 데이터 저장
        result = save_json(self.test_data, json_path)
        
        # 저장 성공 확인
        self.assertTrue(result)
        self.assertTrue(os.path.exists(json_path))
        
        # 파일 로드
        loaded_data = load_json(json_path)
        
        # 원본 데이터와 로드된 데이터 비교
        self.assertEqual(loaded_data, self.test_data)
        
        # 존재하지 않는 파일 로드 테스트
        empty_data = load_json(os.path.join(self.test_dir, 'non_existent.json'))
        self.assertEqual(empty_data, {})
    
    def test_get_timestamp(self):
        """get_timestamp 함수 테스트"""
        # 타임스탬프 생성
        timestamp = get_timestamp()
        
        # 포맷 확인 (YYYYMMDD_HHMMSS)
        self.assertRegex(timestamp, r'^\d{8}_\d{6}$')
        
        # 현재 시간과 근접한지 확인
        current_time = datetime.now()
        timestamp_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
        time_diff = (current_time - timestamp_time).total_seconds()
        
        # 테스트 실행 시간을 고려하여 10초 이내 차이를 허용
        self.assertTrue(time_diff >= 0)
        self.assertTrue(time_diff < 10)
    
    def test_create_experiment_dir(self):
        """create_experiment_dir 함수 테스트"""
        # 테스트 베이스 디렉토리
        base_dir = os.path.join(self.test_dir, 'experiments')
        
        # 지정된 이름으로 실험 디렉토리 생성
        exp_name = "test_experiment"
        exp_dir = create_experiment_dir(base_dir, exp_name)
        
        # 경로 확인
        expected_path = os.path.join(base_dir, exp_name)
        self.assertEqual(exp_dir, expected_path)
        self.assertTrue(os.path.exists(exp_dir))
        self.assertTrue(os.path.isdir(exp_dir))
        
        # 이름을 지정하지 않고 생성
        auto_exp_dir = create_experiment_dir(base_dir)
        
        # 자동 생성된 디렉토리 이름이 'experiment_' 접두어로 시작하는지 확인
        self.assertTrue(os.path.exists(auto_exp_dir))
        self.assertRegex(os.path.basename(auto_exp_dir), r'^experiment_\d{8}_\d{6}$')
    
    def test_format_time(self):
        """format_time 함수 테스트"""
        # 초 단위 테스트
        self.assertEqual(format_time(5.3), "5.3초")
        self.assertEqual(format_time(45.7), "45.7초")
        
        # 분 단위 테스트
        self.assertEqual(format_time(65), "1분 5초")
        self.assertEqual(format_time(125), "2분 5초")
        
        # 시간 단위 테스트
        self.assertEqual(format_time(3665), "1시간 1분 5초")
        self.assertEqual(format_time(7325), "2시간 2분 5초")
        self.assertEqual(format_time(36000), "10시간 0분 0초")
    
    def test_convert_to_serializable(self):
        """convert_to_serializable 함수 테스트"""
        # NumPy 타입 테스트
        self.assertEqual(convert_to_serializable(np.int32(42)), 42)
        
        # float32는 약간의 오차가 있을 수 있으므로 거의 같은지 확인
        float_val = convert_to_serializable(np.float32(3.14))
        self.assertIsInstance(float_val, float)
        self.assertAlmostEqual(float_val, 3.14, places=5)
        self.assertEqual(convert_to_serializable(np.array([1, 2, 3])), [1, 2, 3])
        
        # Pandas 타입 테스트
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        expected_df = [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
        self.assertEqual(convert_to_serializable(df), expected_df)
        
        series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        expected_series = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(convert_to_serializable(series), expected_series)
        
        # 중첩 구조 테스트
        complex_obj = {
            'array': np.array([1, 2, 3]),
            'int': np.int64(42),
            'float': np.float64(3.14),
            'df': df,
            'series': series,
            'list': [np.int32(1), np.float32(2.5)],
            'set': {np.int64(1), np.int64(2)},
            'tuple': (np.int64(1), np.float64(2.5))
        }
        
        converted = convert_to_serializable(complex_obj)
        
        # 변환된 객체는 기본 Python 타입만 포함해야 함
        json_str = json.dumps(converted)  # JSON 직렬화 성공 여부로 확인
        self.assertTrue(isinstance(json_str, str))
        
        # 일부 항목 검사
        self.assertEqual(converted['array'], [1, 2, 3])
        self.assertEqual(converted['int'], 42)
        self.assertEqual(converted['float'], 3.14)
        self.assertEqual(converted['df'], expected_df)


if __name__ == '__main__':
    unittest.main()
