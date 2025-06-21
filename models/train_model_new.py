#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
주어진 피처 데이터를 사용하여 예측 모델을 학습하고,
교차 검증을 통해 성능을 평가한 후, 최종 모델을 저장합니다.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 모듈 임포트
from models.base import BaseLogger, ConfigManager
from models.data_loader import DataLoader
from models.training import XGBoostModel
from models.evaluation import ModelEvaluator
from models.utils import create_experiment_dir, format_time, save_json, get_timestamp

# 설정 로드
try:
    from config.config import PATH_PARAMS, DATA_PARAMS, MODEL_PARAMS
except ImportError:
    print("설정 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
    # 기본 설정
    PATH_PARAMS = {
        'data_path': os.path.join(project_root, 'data'),
        'model_path': os.path.join(project_root, 'models', 'saved'),
        'log_path': os.path.join(project_root, 'logs'),
        'result_path': os.path.join(project_root, 'results')
    }
    DATA_PARAMS = {
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'train_test_split': 0.8,
        'random_state': 42
    }
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 10,
            'verbose_eval': 10
        }
    }


def parse_arguments():
    """
    명령줄 인수를 파싱합니다.
    """
    parser = argparse.ArgumentParser(description='모델 학습 및 평가 스크립트')
    parser.add_argument('--symbol', type=str, default=None,
                        help='트레이딩 심볼 (예: BTC/USDT)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--model-path', type=str, default=None,
                        help='모델 저장 디렉토리 경로')
    parser.add_argument('--no-cv', action='store_true',
                        help='교차 검증 건너뛰기 (기본: False)')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='교차 검증 폴드 수 (기본: 5)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='실험 이름 (기본: timestamp 기반)')
    
    return parser.parse_args()


def setup_paths(args) -> Dict[str, str]:
    """
    경로 설정을 구성합니다.
    
    Args:
        args: 명령줄 인수
        
    Returns:
        경로 설정 딕셔너리
    """
    paths = PATH_PARAMS.copy()
    
    # 디버깅: PATH_PARAMS 출력
    print("PATH_PARAMS:", PATH_PARAMS)
    print("paths 초기값:", paths)
    
    # 명령줄 인수로 지정된 경로가 있으면 업데이트
    if args.data_path:
        paths['data_path'] = args.data_path
    if args.model_path:
        paths['model_path'] = args.model_path
    
    # 필요한 디렉토리 생성
    for path_name, path in paths.items():
        os.makedirs(path, exist_ok=True)
    
    # result_path가 없으면 추가
    if 'result_path' not in paths:
        paths['result_path'] = os.path.join(project_root, 'results')
        os.makedirs(paths['result_path'], exist_ok=True)
        print("result_path 추가됨:", paths['result_path'])
    
    print("최종 paths:", paths)
    return paths


def setup_experiment(paths: Dict[str, str], experiment_name: str = None) -> str:
    """
    실험을 위한 디렉토리를 설정합니다.
    
    Args:
        paths: 경로 설정 딕셔너리
        experiment_name: 실험 이름
        
    Returns:
        실험 디렉토리 경로
    """
    print("setup_experiment 함수 진입, paths:", paths)
    
    if experiment_name is None:
        experiment_name = f"experiment_{get_timestamp()}"
    
    # paths에 result_path가 없으면 추가
    if not paths or 'result_path' not in paths:
        print("Warning: paths가 None이거나 result_path가 없습니다.")
        result_path = os.path.join(project_root, 'results')
        os.makedirs(result_path, exist_ok=True)
    else:
        result_path = paths['result_path']
    
    # 직접 실험 디렉토리 생성
    experiment_dir = os.path.join(result_path, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"생성된 experiment_dir: {experiment_dir}")
    return experiment_dir


def main():
    # 시작 시간
    start_time = time.time()
    
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 로거 초기화
    logger = BaseLogger("ModelTraining")
    logger.info("="*50)
    logger.info("모델 학습 파이프라인 시작")
    logger.info("="*50)
    
    try:
        # 경로 설정
        paths = setup_paths(args)
        logger.info(f"설정된 경로: {paths}")
        
        # 실험 디렉토리 설정
        experiment_dir = setup_experiment(paths, args.experiment_name)
        logger.info(f"실험 결과 저장 경로: {experiment_dir}")
        
        # 심볼 설정
        symbol = args.symbol or DATA_PARAMS.get('symbol', 'BTC/USDT')
        symbol_filename = symbol.replace('/', '').lower()
        
        # 데이터 파일 경로
        train_data_filename = f"{symbol_filename}_labeled_features_train.parquet"
        train_data_path = os.path.join(paths['data_path'], 'processed', train_data_filename)
        
        # 데이터 경로 및 파일 존재 여부 확인
        logger.info("\n" + "="*50)
        logger.info("데이터 경로 확인:")
        logger.info(f"- data_path: {paths['data_path']}")
        processed_dir = os.path.join(paths['data_path'], 'processed')
        processed_dir_exists = os.path.exists(processed_dir)
        logger.info(f"- processed 디렉토리 존재: {processed_dir_exists}")
        logger.info(f"- 전체 학습 데이터 경로: {train_data_path}")
        file_exists = os.path.exists(train_data_path)
        logger.info(f"- 파일 존재 여부: {file_exists}")
        
        # 디렉토리 내용 확인
        if processed_dir_exists:
            processed_files = os.listdir(processed_dir)
            logger.info(f"\nprocessed 디렉토리 내용 ({len(processed_files)}개):")
            for f in sorted(processed_files)[:10]:  # 처음 10개 파일만 출력
                logger.info(f"- {f}")
            if len(processed_files) > 10:
                logger.info(f"- ...외 {len(processed_files)-10}개 파일 더 있음")
        logger.info("="*50 + "\n")
        
        if not file_exists:
            logger.error(f"학습 데이터 파일을 찾을 수 없습니다: {train_data_path}")
            sys.exit(1)
        
        # 1. 데이터 로드
        logger.info("1. 데이터 로드 시작")
        data_loader = DataLoader(logger=logger)
        df_train = data_loader.load(train_data_path)
        
        # 2. 데이터 전처리
        logger.info("2. 데이터 전처리 시작")
        df_preprocessed = data_loader.preprocess(df_train)
        
        # 3. 특성-타겟 분리
        logger.info("3. 특성-타겟 분리 시작")
        X_train, y_train = data_loader.split_features_target(df_preprocessed)
        
        if X_train.empty or len(y_train) == 0:
            logger.error("유효한 학습 데이터가 없습니다.")
            sys.exit(1)
        
        # 4. 모델 초기화
        logger.info("4. 모델 초기화")
        
        # XGBoost 모델 파라미터 가져오기 (MODEL_PARAMS에 이미 early_stopping_rounds가 포함됨)
        model_params = MODEL_PARAMS.get('xgboost', {})
        model = XGBoostModel(model_params, logger=logger)
        evaluator = ModelEvaluator(logger=logger)
        
        # 5. 교차 검증 (선택적)
        if not args.no_cv:
            logger.info("5. 교차 검증 시작")
            cv_results = evaluator.cross_validate(
                model, X_train, y_train, 
                n_splits=args.n_splits
            )
            
            # 교차 검증 결과 저장
            cv_results_path = os.path.join(experiment_dir, "cv_results.json")
            save_json(cv_results, cv_results_path)
            logger.info(f"교차 검증 결과 저장 완료: {cv_results_path}")
        
        # 6. 최종 모델 학습
        logger.info("6. 최종 모델 학습 시작")
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.model_selection import train_test_split
        
        # 학습용/검증용 데이터 분할 (20%를 검증용으로 사용)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"XGBoost 학습용 데이터 크기: {X_train_split.shape}, 검증용 데이터 크기: {X_val_split.shape}")
        
        # 클래스 가중치 계산
        sample_weights = compute_sample_weight('balanced', y_train_split)
        
        # 모델 학습 (검증 데이터셋 추가)
        # early_stopping_rounds는 이미 모델 초기화 시 지정하였음
        final_model = model.train(
            X_train_split, 
            y_train_split, 
            sample_weight=sample_weights,
            eval_set=[(X_val_split, y_val_split)],
            verbose=10
        )
        
        # 7. 특성 중요도 계산
        logger.info("7. 특성 중요도 분석")
        feature_importance = model.get_feature_importance()
        
        # 상위 20개 특성 출력
        logger.info("\n상위 20개 중요 특성:")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            logger.info(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # 특성 중요도 시각화
        importance_plot_path = os.path.join(experiment_dir, "feature_importance.png")
        evaluator.plot_feature_importance(
            feature_importance, 
            top_n=20, 
            save_path=importance_plot_path
        )
        
        # 8. 모델 저장
        logger.info("8. 모델 저장")
        model_dir = os.path.join(paths['model_path'], get_timestamp())
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "final_model.joblib")
        model.save(model_path)
        logger.info(f"모델 저장 완료: {model_path}")
        
        # 파라미터 저장
        params_path = os.path.join(model_dir, "model_params.json")
        save_json(MODEL_PARAMS['xgboost'], params_path)
        logger.info(f"모델 파라미터 저장 완료: {params_path}")
        
        # 특성 중요도 저장
        importance_csv_path = os.path.join(model_dir, "feature_importance.csv")
        feature_importance.to_csv(importance_csv_path, index=False)
        logger.info(f"특성 중요도 저장 완료: {importance_csv_path}")
        
        # 소요 시간 계산 및 출력
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*50)
        logger.info(f"모델 학습 파이프라인 완료! 소요 시간: {format_time(elapsed_time)}")
        logger.info("="*50)
        
    except Exception as e:
        import traceback
        logger.error(f"모델 학습 파이프라인 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
