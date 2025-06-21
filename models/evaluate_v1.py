# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import joblib
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- 설정 파일 임포트 ---
try:
    from config.config import PATH_PARAMS
except ImportError as e:
    print("오류: 설정을 가져올 수 없습니다. 'config/config.py' 파일이 올바른지 확인하세요.")
    print(f"상세 정보: {e}")
    sys.exit(1)

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- 핵심 기능 함수 ---
def prepare_data(df, feature_names=None):
    """평가를 위해 데이터를 준비합니다. 레이블을 매핑하고 피처와 타겟을 분리합니다."""
    logger.info("평가를 위한 데이터 준비 중...")
    # label 컬럼이 없으면 bin 컬럼을 레이블로 사용
    label_col = None
    if 'label' in df.columns:
        label_col = 'label'
    elif 'bin' in df.columns:
        label_col = 'bin'
        logger.warning("'label' 컬럼이 없어 'bin' 컬럼을 레이블로 사용합니다.")
    else:
        logger.error("데이터에서 'label' 또는 'bin' 컬럼을 찾을 수 없습니다.")
        return None, None

    # 타겟 변수 매핑: [-1, 0, 1] -> [0, 1, 2] (bin 컬럼은 이미 0/1/2일 수 있음)
    if label_col == 'label':
        df['target'] = df[label_col].map({-1: 0, 0: 1, 1: 2})
    else:
        df['target'] = df[label_col]
    # 유효하지 않은 타겟이 있는 행 제거 및 정수 타입 확인
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)
    y = df['target']
    # 피처 컬럼 선택 (식별자 및 레이블 컬럼 제외)
    features_to_exclude = ['label', 'bin', 'target', 'timestamp', 't_event_end']
    features = [col for col in df.columns if col not in features_to_exclude]
    X = df[features].copy()
    # 만약 feature_names가 주어지면, 해당 피처만 남김 (모델 학습 피처와 일치)
    if feature_names is not None:
        X = X[[col for col in feature_names if col in X.columns]]
    # 중요: 피처에 NaN 값이 없는지 확인하고, 타겟 인덱스 정렬
    if X.isnull().values.any():
        logger.warning("피처에서 NaN 값을 발견했습니다. 해당 행을 제거합니다.")
        X.dropna(inplace=True)
        y = y.loc[X.index]
    logger.info(f"데이터 준비 완료. X 형태: {X.shape}, y 형태: {y.shape}")
    return X, y

def evaluate_model(model, X_test, y_test):
    """모델을 평가하고 성능 지표를 출력합니다."""
    logger.info("--- 모델 평가 시작 ---")
    
    # 예측 생성
    y_pred = model.predict(X_test)
    
    # 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 결과 출력
    print("\n" + "="*30)
    print("   모델 성능 리포트")
    print("="*30)
    logger.info(f"정확도 (Accuracy):  {accuracy:.4f}")
    logger.info(f"정밀도 (Precision): {precision:.4f} (가중 평균)")
    logger.info(f"재현율 (Recall):    {recall:.4f} (가중 평균)")
    logger.info(f"F1-점수 (F1-Score):  {f1:.4f} (가중 평균)")
    logger.info(f"혼동 행렬 (Confusion Matrix):\n{conf_matrix}")
    print("="*30 + "\n")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'confusion_matrix': conf_matrix}

def plot_feature_importance(model, X):
    """피처 중요도 상위 10개 바 플롯 시각화"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=fi_df.head(10), palette='viridis')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.show()
        return fi_df
    else:
        print('이 모델은 feature_importances_ 속성을 지원하지 않습니다.')
        return None

def plot_feature_distributions(df, top_features):
    """상위 중요도 피처별 레이블(0/1/2) 분포 박스플롯"""
    for col in top_features:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='target', y=col, data=df, palette='Set2')
            plt.title(f'{col} distribution by label (target)')
            plt.tight_layout()
            plt.show()

def main():
    """평가 스크립트의 메인 실행 함수"""
    try:
        # 1. 모델 로드
        model_path = os.path.join(PATH_PARAMS['model_path'], 'model_final.joblib')
        if not os.path.exists(model_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return
        model = joblib.load(model_path)
        logger.info(f"모델 로드 성공: {model_path}")

        # 2. 테스트 데이터 로드
        test_data_path = os.path.join(PATH_PARAMS['data_path'], 'processed', 'btcusdt_labeled_features_test.parquet')
        if not os.path.exists(test_data_path):
            logger.error(f"테스트 데이터 파일을 찾을 수 없습니다: {test_data_path}")
            return
        df_test = pd.read_parquet(test_data_path)
        logger.info(f"테스트 데이터 로드 성공. 형태: {df_test.shape}")

        # 3. 데이터 준비 (모델 학습 피처와 동일하게 맞춤)
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            feature_names = model.get_booster().feature_names
        X_test, y_test = prepare_data(df_test, feature_names=feature_names)
        
        if X_test is None or y_test is None or X_test.empty:
            logger.error("테스트 데이터 준비 실패. 평가를 중단합니다.")
            return
            
        logger.info(f"테스트 데이터 준비 완료. X_test 형태: {X_test.shape}, y_test 형태: {y_test.shape}")

        # 4. 모델 평가
        evaluate_model(model, X_test, y_test)

        # 5. Feature Importance 시각화
        fi_df = plot_feature_importance(model, X_test)
        if fi_df is not None:
            # 6. 상위 중요도 피처별 분포 시각화 (최대 5개)
            top_features = fi_df['feature'].head(5).tolist()
            # target 컬럼이 없으면 생성
            if 'target' not in df_test.columns:
                df_test['target'] = df_test['label'].map({-1: 0, 0: 1, 1: 2})
            plot_feature_distributions(df_test, top_features)

        # 7. Confusion Matrix 및 Classification Report 시각화/출력
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, digits=3, output_dict=True)
        print("\n[Confusion Matrix Table]")
        print(pd.DataFrame(cm))
        print("\n[Detailed Classification Report]")
        print(pd.DataFrame(class_report).T)
        # 클래스별 실수 유형 분석
        print("\n[Error Analysis]")
        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                if i != j and val > 0:
                    print(f"실제 {i} → 예측 {j}: {val}회")
        # 가장 많이 혼동되는 클래스 쌍
        off_diag = [(i, j, cm[i, j]) for i in range(len(cm)) for j in range(len(cm)) if i != j]
        if off_diag:
            most_confused = max(off_diag, key=lambda x: x[2])
            print(f"\n가장 많이 혼동되는 클래스: 실제 {most_confused[0]} → 예측 {most_confused[1]} ({most_confused[2]}회)")
        # 정밀도 vs 재현율 비교
        print("\n[Precision vs Recall by Class]")
        for k in class_report.keys():
            if k in ['0', '1', '2', '-1']:
                p = class_report[k]['precision']
                r = class_report[k]['recall']
                print(f"Class {k}: Precision={p:.3f}, Recall={r:.3f}, {'정밀도 우위' if p>r else '재현율 우위' if r>p else '동일'}")

    except Exception as e:
        logger.error(f"평가 중 예기치 않은 오류 발생: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
