import pandas as pd
import os

# 파일 경로 설정
file_path = 'data/processed/labeled_btcusdt_data.parquet'

# 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
else:
    # 파일 크기 확인
    file_size = os.path.getsize(file_path)
    print(f"파일 경로: {file_path}")
    print(f"파일 크기: {file_size} 바이트")
    
    # Parquet 파일 읽기
    try:
        df = pd.read_parquet(file_path)
        
        # 기본 정보 출력
        print("\n=== 데이터프레임 정보 ===")
        print(f"전체 행 수: {len(df)}")
        print("\n=== 컬럼 목록 ===")
        print(df.columns.tolist())
        
        # 레이블 분포 확인
        if 'bin' in df.columns:
            print("\n=== 레이블 분포 ===")
            label_dist = df['bin'].value_counts().sort_index()
            print(label_dist)
            
            # 레이블 비율 계산
            if not label_dist.empty:
                print("\n=== 레이블 비율 ===")
                print((label_dist / len(df) * 100).round(1).astype(str) + '%')
        
        # 상위 5개 행 출력
        print("\n=== 상위 5개 행 ===")
        print(df.head())
        
        # 결측치 확인
        print("\n=== 결측치 개수 ===")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"\n파일을 읽는 중 오류 발생: {str(e)}")
        # 오류 발생 시 상세 정보 출력
        import traceback
        traceback.print_exc()

# 추가로 다른 레이블 파일도 확인
test_file = 'data/processed/btcusdt_labeled_features_test.parquet'
if os.path.exists(test_file):
    print(f"\n\n=== 테스트 파일 확인: {test_file} ===")
    try:
        test_df = pd.read_parquet(test_file)
        print(f"테스트 데이터 행 수: {len(test_df)}")
        if 'bin' in test_df.columns:
            print("\n테스트 데이터 레이블 분포:")
            print(test_df['bin'].value_counts().sort_index())
    except Exception as e:
        print(f"테스트 파일 읽기 오류: {str(e)}")
