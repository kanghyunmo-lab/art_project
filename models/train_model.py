# -*- coding: utf-8 -*-
"""
주어진 피처 데이터를 사용하여 예측 모델을 학습하고,
교차 검증을 통해 성능을 평가한 후, 최종 모델을 저장합니다.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from xgboost.callback import EarlyStopping

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from config import PATH_PARAMS, DATA_PARAMS, MODEL_PARAMS

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Parquet 파일에서 학습 데이터를 로드합니다."""
    if not os.path.exists(file_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    logger.info(f"학습 데이터 로드 중: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"데이터 로드 완료. Shape: {df.shape}")
    return df

def prepare_data(df: pd.DataFrame):
    """모델 학습을 위해 데이터를 준비합니다 (피처와 타겟 분리)."""
    # 정보 누수 가능성이 있는 컬럼 및 불필요한 컬럼 제거
    cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'ret', 'trgt', 'bin']
    
    # 't1' 컬럼은 레이블링 과정에서 생성되며, 모델 학습 시에는 사용하지 않음
    if 't1' in df.columns:
        cols_to_drop.append('t1')

    # 존재하는 컬럼만 삭제 리스트에 포함
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=cols_to_drop_existing)
    y = df['bin']

    # --- 디버깅 시작 ---
    logger.info(f"레이블 매핑 전 y 고유값: {sorted(y.unique())}")
    if y.isnull().any():
        logger.warning(f"y에 NaN 값이 {y.isnull().sum()}개 존재합니다. 중립(0)으로 채웁니다.")
        y = y.fillna(0)
    # --- 디버깅 끝 ---

    # XGBoost 레이블 요구사항에 맞게 타겟 변수 매핑 (-1, 0, 1 -> 0, 1, 2)
    y = y.map({-1: 0, 0: 1, 1: 2}).astype(int)
    logger.info("타겟 변수(y)를 XGBoost 요구사항에 맞게 [0, 1, 2]로 매핑했습니다.")
    logger.info(f"레이블 매핑 후 y 고유값: {sorted(y.unique())}")

    # XGBoost가 처리할 수 없는 datetime 타입 컬럼 제거
    X = X.select_dtypes(exclude=['datetime64[ns, UTC]', 'datetime64[ns]'])
    
    logger.info(f"피처(X)와 타겟(y) 분리 완료.")
    logger.info(f"사용된 피처 개수: {len(X.columns)}")
    logger.debug(f"피처 목록: {X.columns.tolist()}")
    
    return X, y

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """교차 검증을 통해 모델을 학습하고 평가합니다."""
    
    # 클래스 불균형 처리는 다중 분류의 경우 sample_weight 등 다른 방식을 고려해야 함
    # scale_pos_weight는 이진 분류용이므로 여기서는 사용하지 않음
    
    # 모델 파라미터 로드
    model_params = MODEL_PARAMS['xgboost'].copy()

    model = xgb.XGBClassifier(**model_params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    logger.info("교차 검증 시작 (5-fold)...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        early_stopping_callback = EarlyStopping(rounds=50, save_best=False)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping_callback],
                  verbose=False)
        
        preds = model.predict(X_val)
        
        metrics['accuracy'].append(accuracy_score(y_val, preds))
        metrics['precision'].append(precision_score(y_val, preds, average='weighted', zero_division=0))
        metrics['recall'].append(recall_score(y_val, preds, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(y_val, preds, average='weighted', zero_division=0))
        
        logger.info(f"  Fold {fold+1}/5 - Accuracy: {metrics['accuracy'][-1]:.4f}, F1: {metrics['f1'][-1]:.4f}")

    logger.info("-" * 30)
    logger.info("교차 검증 평균 성능:")
    logger.info(f"  - Accuracy: {np.mean(metrics['accuracy']):.4f}")
    logger.info(f"  - Precision: {np.mean(metrics['precision']):.4f}")
    logger.info(f"  - Recall: {np.mean(metrics['recall']):.4f}")
    logger.info(f"  - F1-Score: {np.mean(metrics['f1']):.4f}")
    logger.info("-" * 30)
    
    # 전체 학습 데이터로 최종 모델 학습
    logger.info("전체 학습 데이터로 최종 모델 학습 시작...")
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X, y, verbose=False)
    logger.info("최종 모델 학습 완료.")
    
    return final_model

def save_model(model, file_path: str):
    """학습된 모델을 파일로 저장합니다."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"모델이 성공적으로 저장되었습니다: {file_path}")
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {e}")

if __name__ == '__main__':
    try:
        # 1. 데이터 로드
        symbol = DATA_PARAMS.get('symbol', 'BTC/USDT').replace('/', '').lower()
        train_data_filename = f"{symbol}_labeled_features_train.parquet"
        train_data_path = os.path.join(PATH_PARAMS['data_path'], 'processed', train_data_filename)
        
        df_train = load_data(train_data_path)
        
        if not df_train.empty:
            # 2. 데이터 준비
            X_train, y_train = prepare_data(df_train)
            
            # 3. 모델 학습 및 평가
            final_model = train_and_evaluate(X_train, y_train)
            
            # 4. 모델 저장
            model_save_path = os.path.join(PATH_PARAMS['model_path'], 'model_final.joblib')
            save_model(final_model, model_save_path)
            
    except Exception as e:
        logger.error(f"모델 학습 파이프라인 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
