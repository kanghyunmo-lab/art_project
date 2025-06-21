# -*- coding: utf-8 -*-
"""
주어진 피처 데이터를 사용하여 예측 모델을 학습하고,
교차 검증을 통해 성능을 평가한 후, 최종 모델을 저장합니다.
"""
import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
import traceback
from typing import Dict, Tuple, Optional, List, Any, Union
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from xgboost.callback import EarlyStopping

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from config.config import PATH_PARAMS, DATA_PARAMS, MODEL_PARAMS

# 로깅 설정
def setup_logging():
    """로깅 설정을 초기화하고 로거를 반환합니다."""
    # 로그 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(current_dir), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 경로
    log_file = os.path.join(log_dir, 'model_training.log')
    
    # 로거 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 로깅 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 포매터 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"로깅이 설정되었습니다. 로그 파일: {log_file}")
    return logger

# 로거 초기화
logger = setup_logging()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Parquet 파일에서 학습 데이터를 로드하고 기본 전처리를 수행합니다.
    
    Args:
        file_path: 로드할 Parquet 파일 경로
        
    Returns:
        전처리된 DataFrame
    """
    if not os.path.exists(file_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    
    logger.info(f"학습 데이터 로드 중: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"데이터 로드 완료. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

def prepare_data(df: pd.DataFrame):
    """
    모델 학습을 위해 데이터를 준비합니다.
    - 불필요한 컬럼 제거
    - 클래스 불균형 처리
    - 피처와 타겟 분리
    
    Args:
        df: 전처리할 DataFrame
        
    Returns:
        tuple: (X, y) - 특성 행렬과 타겟 벡터
    """
    if df.empty:
        logger.error("입력 데이터프레임이 비어 있습니다.")
        return None, None
    
    # 1. 불필요한 컬럼 제거
    cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'ret', 'trgt', 'bin', 't1']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # 2. 타겟 변수 추출
    y = df['bin'].copy()
    
    # 3. NaN 처리
    if y.isnull().any():
        logger.warning(f"y에 NaN 값이 {y.isnull().sum()}개 존재합니다. 중립(0)으로 채웁니다.")
        y = y.fillna(0)
    
    # 4. 클래스 매핑 (-1, 0, 1 -> 0, 1, 2)
    y = y.map({-1: 0, 0: 1, 1: 2}).astype(int)
    
    # 5. 클래스 분포 분석
    class_dist = y.value_counts().sort_index()
    class_names = {0: '매도', 1: '중립', 2: '매수'}
    
    logger.info("\n" + "="*50)
    logger.info("클래스 분포:")
    for cls, count in class_dist.items():
        cls_name = class_names.get(cls, str(cls))
        logger.info(f"{cls_name} ({cls}): {count}개 ({count/len(y)*100:.2f}%)")
    logger.info("="*50 + "\n")
    
    # 6. 클래스 불균형 처리 (언더샘플링)
    min_samples = class_dist.min()
    if min_samples < 100:  # 최소 샘플 수가 100개 미만이면 언더샘플링
        logger.warning("클래스 불균형이 심각합니다. 언더샘플링을 수행합니다.")
        
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
        df = df.loc[idx_list].copy()
        y = y.loc[idx_list].copy()
        
        logger.info(f"언더샘플링 후 데이터 크기: {len(df)}")
        logger.info("새로운 클래스 분포:")
        for cls, count in y.value_counts().sort_index().items():
            cls_name = class_names.get(cls, str(cls))
            logger.info(f"{cls_name} ({cls}): {count}개 ({count/len(y)*100:.2f}%)")
    
    # 7. 피처 행렬 준비
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 8. datetime 타입 컬럼 제거
    X = X.select_dtypes(exclude=['datetime64[ns, UTC]', 'datetime64[ns]'])
    
    # 9. 최종 데이터 검증
    if len(X) == 0 or len(y) == 0:
        logger.error("유효한 데이터가 없습니다.")
        return None, None
    
    logger.info(f"\n최종 데이터 크기: {len(X)}개 샘플, {X.shape[1]}개 특성")
    logger.debug(f"사용된 피처 목록: {X.columns.tolist()}")
    
    return X, y

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    교차 검증을 통해 모델을 학습하고 평가합니다.
    
    Args:
        X: 학습 데이터 피처
        y: 타겟 레이블 (0: 매도, 1: 중립, 2: 매수)
        
    Returns:
        tuple: (best_model, feature_importance_df) - 최적 모델과 특성 중요도 데이터프레임
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        logger.error("유효한 학습 데이터가 없습니다.")
        return None, None
    
    # 클래스 이름 매핑
    class_names = {0: '매도', 1: '중립', 2: '매수'}
    
    # 클래스 분포 출력
    class_dist = y.value_counts().sort_index()
    logger.info("\n" + "="*50)
    logger.info("학습 데이터 클래스 분포:")
    for cls, count in class_dist.items():
        cls_name = class_names.get(cls, str(cls))
        logger.info(f"{cls_name} ({cls}): {count}개 ({count/len(y)*100:.2f}%)")
    logger.info("="*50 + "\n")
    
    # 클래스 가중치 계산 (불균형 데이터셋 대응)
    class_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    # 교차 검증 설정 (Stratified K-Fold 사용)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 평가 지표 저장을 위한 딕셔너리
    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }
    
    # 클래스별 지표 저장
    class_metrics = {cls: {'precision': [], 'recall': [], 'f1': []} for cls in class_names.keys()}
    
    # 특성 중요도 저장을 위한 배열
    feature_importances = np.zeros(X.shape[1])
    
    # 모델 초기화
    model_params = MODEL_PARAMS['xgboost'].copy()
    eval_metric = model_params.pop('eval_metric', 'mlogloss')
    
    # 조기 종료 콜백
    early_stop = xgb.callback.EarlyStopping(
        rounds=MODEL_PARAMS.get('early_stopping_rounds', 50),
        save_best=True,
        maximize=False,
        data_name='validation_0',
        metric_name=eval_metric
    )
    
    logger.info(f"교차 검증 시작 (n_splits={n_splits})...")
    
    # 평가 지표 저장을 위한 딕셔너리 초기화
    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }
    
    # 클래스별 지표 저장
    class_metrics = {cls: {'precision': [], 'recall': [], 'f1': []} for cls in class_names.keys()}
    
    # 특성 중요도 저장을 위한 배열
    feature_importances = np.zeros(X.shape[1])
    
    logger.info("\n교차 검증 시작...")
    
    # 교차 검증 수행
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # 데이터 분할
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 클래스 가중치 계산 (클래스 불균형 고려)
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train_fold
        )
        
        # 모델 초기화 - 필요한 파라미터만 명시적으로 설정
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            early_stopping_rounds=10,
            num_class=3,  # 3개 클래스 분류
            verbosity=1  # 로그 레벨 설정
        )
        
        logger.info("\n" + "="*50)
        logger.info(f"Fold {fold_idx}/{n_splits} - 학습 시작")
        logger.info(f"훈련 세트: {len(X_train_fold)}개, 검증 세트: {len(X_val_fold)}개")
        
        try:
            # 모델 학습
            model.fit(
                X_train_fold, 
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                sample_weight=sample_weights,
                verbose=10
            )
            
            # 검증 세트 예측
            y_pred = model.predict(X_val_fold)
            
            # 평가 지표 계산
            acc = accuracy_score(y_val_fold, y_pred)
            prec_macro = precision_score(y_val_fold, y_pred, average='macro')
            rec_macro = recall_score(y_val_fold, y_pred, average='macro')
            f1_macro = f1_score(y_val_fold, y_pred, average='macro')
            prec_weighted = precision_score(y_val_fold, y_pred, average='weighted')
            rec_weighted = recall_score(y_val_fold, y_pred, average='weighted')
            f1_weighted = f1_score(y_val_fold, y_pred, average='weighted')
            
            # 클래스별 지표 계산
            class_report = classification_report(
                y_val_fold, y_pred, 
                target_names=[class_names[i] for i in sorted(class_names.keys())],
                output_dict=True
            )
            
            # 지표 저장
            metrics['accuracy'].append(acc)
            metrics['precision_macro'].append(prec_macro)
            metrics['recall_macro'].append(rec_macro)
            metrics['f1_macro'].append(f1_macro)
            metrics['precision_weighted'].append(prec_weighted)
            metrics['recall_weighted'].append(rec_weighted)
            metrics['f1_weighted'].append(f1_weighted)
            
            # 클래스별 지표 저장 - class_report의 모든 키를 문자열로 변환하여 처리
            for cls in class_metrics.keys():
                # class_report의 모든 키를 문자열로 변환한 딕셔너리 생성
                str_class_report = {str(k): v for k, v in class_report.items()}
                
                # 클래스별 메트릭 저장
                if str(cls) in str_class_report:
                    try:
                        class_metrics[cls]['precision'].append(float(str_class_report[str(cls)]['precision']))
                        class_metrics[cls]['recall'].append(float(str_class_report[str(cls)]['recall']))
                        class_metrics[cls]['f1'].append(float(str_class_report[str(cls)]['f1-score']))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"클래스 {cls} 메트릭 저장 중 오류: {e}")
                        # 기본값 추가
                        class_metrics[cls]['precision'].append(0.0)
                        class_metrics[cls]['recall'].append(0.0)
                        class_metrics[cls]['f1'].append(0.0)
            
            # 폴드별 결과 출력
            logger.info(f"\n=== Fold {fold_idx} 평가 결과 ===")
            logger.info(f"정확도: {acc:.4f}")
            logger.info(f"매크로 평균 - 정밀도: {prec_macro:.4f}, 재현율: {rec_macro:.4f}, F1: {f1_macro:.4f}")
            logger.info(f"가중 평균 - 정밀도: {prec_weighted:.4f}, 재현율: {rec_weighted:.4f}, F1: {f1_weighted:.4f}")
            
            # class_report 디버깅을 위한 로그 추가
            logger.info("\n=== class_report 구조 디버깅 ===")
            logger.info(f"class_report 타입: {type(class_report)}")
            logger.info(f"class_report 키 목록: {list(class_report.keys())}")
            for k, v in class_report.items():
                logger.info(f"- 키: {k} (타입: {type(k)}), 값 타입: {type(v)}")
                if isinstance(v, dict):
                    logger.info(f"  - 하위 키: {list(v.keys())}")
            
            # 클래스별 성능 로깅
            logger.info("\n클래스별 성능:")
            for cls in sorted(class_report.keys()):
                if cls in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                    
                logger.info(f"\n클래스 {cls} (타입: {type(cls)}):")
                try:
                    # cls가 문자열이면 정수로 변환 시도
                    cls_key = str(int(float(cls))) if str(cls).replace('.', '').isdigit() else str(cls)
                    cls_name = class_names.get(int(float(cls)), str(cls))
                    
                    # class_report에서 값 가져오기
                    metrics = class_report.get(cls_key, {}) or class_report.get(str(cls), {})
                    if not metrics:
                        logger.warning(f"  - {cls} 클래스에 대한 메트릭을 찾을 수 없습니다.")
                        continue
                        
                    logger.info(f"  - 정밀도: {metrics.get('precision', 'N/A'):.4f}")
                    logger.info(f"  - 재현율: {metrics.get('recall', 'N/A'):.4f}")
                    logger.info(f"  - F1 점수: {metrics.get('f1-score', 'N/A'):.4f}")
                    logger.info(f"  - 샘플 수: metrics.get('support', 'N/A')")
                    
                except Exception as e:
                    logger.error(f"  - {cls} 클래스 처리 중 오류 발생: {str(e)}")
                    logger.error(f"  - 오류 유형: {type(e).__name__}")
                    logger.error(f"  - 전체 오류: {traceback.format_exc()}")
            
            # 특성 중요도 누적
            feature_importances += model.feature_importances_
            
        except Exception as e:
            logger.error(f"Fold {fold_idx} 학습 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            continue
        
        # 클래스별 성능 출력
        logger.info("\n클래스별 성능:")
        for cls in sorted(class_metrics.keys()):
            logger.info(f"{class_names[cls]} ({cls}):")
            logger.info(f"  정밀도: {class_report[str(cls)]['precision']:.4f}")
            logger.info(f"  재현율: {class_report[str(cls)]['recall']:.4f}")
            logger.info(f"  F1: {class_report[str(cls)]['f1-score']:.4f}")
        
        # 특성 중요도 누적
        feature_importances += model.feature_importances_
        
        # 폴드별 결과 출력
        logger.info(f"[폴드 {fold} 결과]")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Weighted Precision: {prec:.4f}")
        logger.info(f"  Weighted Recall: {rec:.4f}")
        logger.info(f"  Weighted F1: {f1:.4f}")
        logger.info("\n  클래스별 정밀도:")
        logger.info(f"    매도(-1): {prec_per_class[0]:.4f}")
        logger.info(f"    중립(0): {prec_per_class[1]:.4f}")
        logger.info(f"    매수(1): {prec_per_class[2]:.4f}")
        logger.info("\n  클래스별 재현율:")
        logger.info(f"    매도(-1): {rec_per_class[0]:.4f}")
        logger.info(f"    중립(0): {rec_per_class[1]:.4f}")
        logger.info(f"    매수(1): {rec_per_class[2]:.4f}")
    
    # 특성 중요도 평균 계산
    feature_importances /= n_splits
    
    # 특성 중요도 데이터프레임 생성
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # 상위 20개 특성 출력
    logger.info("\n상위 20개 중요 특성:")
    for i, (_, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
        logger.info(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # 전체 데이터로 최종 모델 학습
    logger.info("\n전체 데이터로 최종 모델 학습 중...")
    final_model = xgb.XGBClassifier(**MODEL_PARAMS['xgboost'])
    final_model.fit(
        X, y,
        sample_weight=compute_sample_weight(class_weight='balanced', y=y),
        verbose=MODEL_PARAMS['verbose_eval']
    )
    logger.info("최종 모델 학습 완료.")
    
    return final_model, feature_importance_df

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
        logger.info("="*50)
        logger.info("모델 학습 파이프라인 시작")
        logger.info("="*50)
        
        # 1. 데이터 로드
        symbol = DATA_PARAMS.get('symbol', 'BTC/USDT').replace('/', '').lower()
        train_data_filename = f"{symbol}_labeled_features_train.parquet"
        train_data_path = os.path.join(PATH_PARAMS['data_path'], 'processed', train_data_filename)
        
        # 데이터 경로 및 파일 존재 여부 확인
        logger.info("\n" + "="*50)
        logger.info("데이터 경로 확인:")
        logger.info(f"- data_path: {PATH_PARAMS['data_path']}")
        logger.info(f"- processed 디렉토리 존재: {os.path.exists(os.path.join(PATH_PARAMS['data_path'], 'processed'))}")
        logger.info(f"- 전체 학습 데이터 경로: {train_data_path}")
        logger.info(f"- 파일 존재 여부: {os.path.exists(train_data_path)}")
        
        # 디렉토리 내용 확인
        if os.path.exists(os.path.join(PATH_PARAMS['data_path'], 'processed')):
            processed_files = os.listdir(os.path.join(PATH_PARAMS['data_path'], 'processed'))
            logger.info(f"\nprocessed 디렉토리 내용 ({len(processed_files)}개):")
            for f in sorted(processed_files)[:10]:  # 처음 10개 파일만 출력
                logger.info(f"- {f}")
            if len(processed_files) > 10:
                logger.info(f"- ...외 {len(processed_files)-10}개 파일 더 있음")
        logger.info("="*50 + "\n")
        
        logger.info(f"학습 데이터 로드 시도: {train_data_path}")
        df_train = load_data(train_data_path)
        
        if df_train is None or df_train.empty:
            logger.error("학습 데이터를 로드할 수 없거나 데이터가 비어 있습니다.")
            sys.exit(1)
            
        logger.info(f"로드된 데이터 크기: {df_train.shape}")
        
        # 2. 데이터 준비
        logger.info("\n데이터 전처리 시작...")
        X_train, y_train = prepare_data(df_train)
        
        if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
            logger.error("데이터 전처리 중 오류가 발생했습니다.")
            sys.exit(1)

        # 레이블 분포 확인
        logger.info("\n" + "="*50)
        logger.info("학습에 사용될 레이블 분포:")
        label_counts = y_train.value_counts().sort_index()
        for label, count in label_counts.items():
            logger.info(f"클래스 {label}: {count}개 ({count/len(y_train)*100:.2f}%)")
        logger.info("="*50)

        # 3. 모델 학습 및 평가
        logger.info("\n모델 학습 시작...")
        final_model, feature_importance_df = train_and_evaluate(X_train, y_train)
        
        if final_model is None:
            logger.error("모델 학습에 실패했습니다.")
            sys.exit(1)
        
        # 4. 모델 및 결과 저장
        logger.info("\n학습 결과 저장 중...")
        os.makedirs(PATH_PARAMS['model_path'], exist_ok=True)
        
        try:
            # 모델 저장
            model_save_path = os.path.join(PATH_PARAMS['model_path'], 'model_final.joblib')
            save_model(final_model, model_save_path)
            
            # 특성 중요도 저장
            if feature_importance_df is not None:
                feature_importance_path = os.path.join(PATH_PARAMS['model_path'], 'feature_importance.csv')
                feature_importance_df.to_csv(feature_importance_path, index=False)
                logger.info(f"특성 중요도가 저장되었습니다: {feature_importance_path}")
            
            # 모델 학습 파라미터 저장
            params_path = os.path.join(PATH_PARAMS['model_path'], 'model_params.json')
            with open(params_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(MODEL_PARAMS['xgboost'], f, indent=2, ensure_ascii=False)
            logger.info(f"모델 파라미터가 저장되었습니다: {params_path}")
            
            logger.info("\n" + "="*50)
            logger.info("모델 학습이 성공적으로 완료되었습니다!")
            logger.info("="*50)
            
        except Exception as save_error:
            logger.error(f"모델 저장 중 오류 발생: {save_error}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"모델 학습 파이프라인 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
