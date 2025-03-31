# 똑같은 코드로 Training_merged_data.csv와 vaildation_merged_data.csv 파일을 전처리함
# Training_merged_data.csv와 vaildation_merged_data.csv은 기존 카톡에 있는 파일과는 살짝 다르다
# 두개의 파일 공통적으로 collection_date 열의 형식이 8월 26일로 되어있을텐데 그렇게되면 TCN모델에서 처리하기 적합하지 않은 형식이라고 함
# 따라서 엑셀에서 셀서식을 2025.8.26 형식으로 변환하여 이 데이터 전처리 코드를 진행함.


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

def preprocess_data(input_file: str, npy_dir: str):
    """
    (input_file: str, output_file: str, npy_dir: str): 원래 함수 변수 코드인데 output_file코드는 주석이라 뺌 -민석-
    센서 데이터 전처리 함수
    :param input_file: 원본 CSV 파일 경로
    :param output_file: 전처리된 CSV 파일 저장 경로
    :param npy_dir: 전처리된 numpy 파일(X, y) 저장 폴더 경로
    :return: 시계열 데이터(X), 레이블(y), 스케일러 객체
    """
    # 데이터 불러오기
    print(f"데이터 로드 중: {input_file}")
    df = pd.read_csv(input_file)
    
    # 날짜 및 시간 변환
    df['timestamp'] = pd.to_datetime(df['collection_date'] + ' ' + df['collection_time'])
    df = df.sort_values(by=['device_name', 'timestamp']).reset_index(drop=True)
    
    # 불필요한 컬럼 제거
    df.drop(columns=['collection_date', 'collection_time', 'device_name'], inplace=True)
    
    # 결측치 처리 (이전 값으로 보간)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # 정규화 (MinMaxScaler 적용)
    scaler = MinMaxScaler()
    scaled_features = df.drop(columns=['timestamp', 'state'])  # state는 레이블로 사용
    df[scaled_features.columns] = scaler.fit_transform(scaled_features)
    
    # # 전처리된 데이터 CSV 파일로 저장 (폴더가 없으면 생성)
    # output_dir = os.path.dirname(output_file)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # print(f"전처리된 데이터 CSV 저장 중: {output_file}")
    # df.to_csv(output_file, index=False)
    # CSV로 전처리된 파일을 확인해봤는데 이상 없는것으로 확인됨. 필요하면 주석풀고 실행해보길 바람 -민석- #
    
    # 시계열 윈도우 생성
    sequence_length = 10  # 예측을 위한 과거 데이터 개수
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length].drop(columns=['timestamp', 'state']).values)
        y.append(df.iloc[i+sequence_length]['state'])
    X, y = np.array(X), np.array(y)
    
    # npy 파일로 저장 (폴더가 없으면 생성)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    npy_X_file = os.path.join(npy_dir, "X.npy")
    npy_y_file = os.path.join(npy_dir, "y.npy")
    print(f"전처리된 데이터 npy 저장 중: {npy_X_file} and {npy_y_file}")
    np.save(npy_X_file, X)
    np.save(npy_y_file, y)
    
    return X, y, scaler

if __name__ == '__main__':
    # 파일 경로 지정 (실제 경로에 맞게 수정)
    input_file = "/Users/kwonminseok/Desktop/restart_TCN/R_env/vaild_data/Validation_merged_data.csv"       # 원본 데이터 경로
    #output_file = "/Users/kwonminseok/Desktop/restart_TCN/R_env/train_data/Training_transform_data.csv"  # 전처리된 CSV 데이터 저장 경로
    #전처리 csv파일 저장 코드는 위에서 주석 처리했기 때문에 파일 경로지정 코드도 주석처리함 -민석-
    npy_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/vaild_data/v_npy_files"                       # 전처리된 npy 파일 저장 폴더 경로
    
    # 전처리 함수 실행
    X, y, scaler = preprocess_data(input_file, npy_dir)
    # 원래 코드 (input_file: str, output_file: str, npy_dir: str): ////이유 이하 동문

    # 결과 확인
    print("입력 데이터(X) shape:", X.shape)
    print("레이블(y) shape:", y.shape)
