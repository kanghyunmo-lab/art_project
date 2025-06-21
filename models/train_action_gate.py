"""
액션게이트 모델 학습 스크립트
15분봉과 4시간봉 데이터를 활용한 LightGBM 모델 학습
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의
SYMBOL = 'BTC/USDT'
TIMEFRAMES = ['15m', '4h']
DATA_DIR = os.path.join('..', 'data', 'processed')
MODEL_DIR = os.path.join('..', 'models', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(timeframe):
    """
    지정된 timeframe에 대한 데이터 로드
    """
    filename = f"btcusdt_labeled_features_{timeframe}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath}를 찾을 수 없습니다.")
    
    logger.info(f"{timeframe} 데이터 로드 중...")
    df = pd.read_parquet(filepath)
    
    # 레이블이 NaN이 아닌 행만 필터링
    df = df[df['bin'].notna()]
    
    return df

def preprocess_data(df, timeframe):
    """
    데이터 전처리
    """
    # 타겟 변수 설정 (액션게이트: 거래 발생 여부)
    # bin이 0이면 0(보유), 그 외는 1(거래 발생)
    y = (df['bin'] != 0).astype(int)
    
    # 피처 선택
    exclude_cols = ['bin', 'ret', 't_event_end']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    # 컬럼명에 timeframe 추가 (다중 timeframe 병합 시 구분을 위함)
    X = X.add_suffix(f'_{timeframe}')
    
    return X, y

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    LightGBM 모델 학습
    """
    # SMOTE를 사용한 오버샘플링
    logger.info("SMOTE를 사용한 오버샘플링 수행 중...")
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 모델 파라미터 설정
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 검증 세트가 제공되면 early_stopping 사용
    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    
    logger.info("모델 학습 시작...")
    model = LGBMClassifier(**params, n_estimators=1000)
    
    if eval_set:
        model.fit(
            X_train_res, y_train_res,
            eval_set=eval_set,
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(50)
            ]
        )
    else:
        model.fit(X_train_res, y_train_res)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    모델 평가
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n[Confusion Matrix]")
    logger.info(confusion_matrix(y_test, y_pred))
    
    logger.info("\n[Classification Report]")
    logger.info(classification_report(y_test, y_pred, digits=3))
    
    return y_pred, y_prob

def save_model(model, model_name):
    """
    모델 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}.pkl")
    
    joblib.dump(model, model_path)
    logger.info(f"모델이 저장되었습니다: {model_path}")
    return model_path

def main():
    try:
        # 1. 데이터 로드
        dfs = {}
        for tf in TIMEFRAMES:
            dfs[tf] = load_data(tf)
        
        # 2. 데이터 전처리
        X_list = []
        y_list = []
        
        for tf in TIMEFRAMES:
            X, y = preprocess_data(dfs[tf], tf)
            X_list.append(X)
            if not y_list:  # 첫 번째 y를 기준으로 함
                y_list.append(y)
        
        # 모든 timeframe의 X 결합
        X_combined = pd.concat(X_list, axis=1)
        y = y_list[0]  # y는 동일하므로 첫 번째 사용
        
        # 인덱스 정렬 및 결측치 제거
        X_combined = X_combined.sort_index()
        y = y.sort_index()
        
        # 3. 훈련/검증/테스트 세트 분할
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        logger.info(f"훈련 세트: {X_train.shape[0]}개, 검증 세트: {X_val.shape[0]}개, 테스트 세트: {X_test.shape[0]}개")
        
        # 4. 모델 학습
        model = train_model(X_train, y_train, X_val, y_val)
        
        # 5. 모델 평가
        logger.info("\n[검증 세트 성능]")
        evaluate_model(model, X_val, y_val)
        
        logger.info("\n[테스트 세트 성능]")
        evaluate_model(model, X_test, y_test)
        
        # 6. 모델 저장
        save_model(model, 'action_gate_model')
        
        logger.info("액션게이트 모델 학습이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
