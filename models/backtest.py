import os
import pandas as pd
import joblib
import logging
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from config.config import PATH_PARAMS, MODEL_PARAMS
def prepare_data_for_local_testing(df):
    """
    백테스트를 위해 데이터를 준비합니다. train_model.py의 함수를 로컬에서 구현합니다.
    레이블 'label'을 타겟 'target'으로 변환하고 피처와 타겟을 분리합니다.
    """
    logger.info("Preparing data for local testing...")
    if 'label' not in df.columns:
        logger.error("Target column 'label' not found.")
        return None, None

    # 다중 클래스 분류를 위해 타겟 변수 매핑: [-1, 0, 1] -> [0, 1, 2]
    df['target'] = df['label'].map({-1: 0, 0: 1, 1: 2})
    
    # 유효한 타겟이 없는 행 제거
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    y = df['target']
    
    # 피처 컬럼 선택 (레이블 관련 및 식별자 컬럼 제외)
    features = [col for col in df.columns if col not in ['label', 'target']]
    X = df[features].copy()
    
    # 피처 데이터에 NaN 값이 있으면 안 됨
    X.dropna(inplace=True)
    y = y.loc[X.index] # NaN 제거 후 인덱스 정렬

    logger.info(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """모델 성능을 평가하고 결과를 출력합니다."""
    logger.info("--- Starting Model Evaluation ---")
    
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 결과 출력
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

def main():
    """백테스트 메인 실행 함수"""
    try:
        # 1. 모델 로드
        model_path = os.path.join(PATH_PARAMS['model_path'], 'model_final.joblib')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")

        # 2. 테스트 데이터 로드
        test_data_path = os.path.join(PATH_PARAMS['data_path'], 'processed', 'btcusdt_labeled_features_test.parquet')
        if not os.path.exists(test_data_path):
            logger.error(f"Test data file not found at: {test_data_path}")
            return
        df_test = pd.read_parquet(test_data_path)
        logger.info(f"Test data loaded successfully from {test_data_path}. Shape: {df_test.shape}")

        # 3. 데이터 준비 (학습 때와 동일한 전처리 적용)
        X_test, y_test = prepare_data_for_local_testing(df_test)
        
        if X_test is None or y_test is None:
            logger.error("Failed to prepare test data. It might not contain valid labels.")
            return
            
        logger.info(f"Test data prepared. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # 4. 모델 평가
        evaluate_model(model, X_test, y_test)

    except Exception as e:
        logger.error(f"An unexpected error occurred during backtesting: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
