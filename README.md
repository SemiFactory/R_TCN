# TCN 시계열 예측 프로젝트

이 프로젝트는 Temporal Convolutional Network(TCN)를 사용하여 시계열 데이터를 기반으로 상태를 예측하는 분류 모델을 구현합니다.

## 프로젝트 구조

```
.
├── TCNModel_S.py          # TCN 모델 정의
├── tcn_train_model_S.py   # 모델 학습 스크립트
├── evaluate_and_pridict_S.py  # 모델 평가 및 예측
├── visual_result_S.py     # 결과 시각화
├── data_preprocessing.py  # 데이터 전처리 모듈

```

## 주요 기능

- 시계열 데이터 기반 상태 예측 (4개 클래스)
- TCN 모델을 사용한 시계열 처리
- 모델 학습 및 평가
- 예측 결과 시각화

## 데이터

프로젝트는 다음 특성을 포함하는 시계열 데이터를 사용합니다:
- PM10, PM2.5, PM1.0 (미세먼지 관련)
- NTC, CT1, CT2, CT3, CT4 (센서 데이터)
- temp_max, ex_temperature, ex_humidity, ex_illuminance (환경 데이터)

## 사용 방법

1. 데이터 준비: `data_preprocessing.py`를 사용하여 데이터 전처리
2. 모델 학습: `TCN_model.py` 실행
3. 모델 평가: `evaluate_and_pridict_S.py` 실행
4. 결과 시각화: `visual_result_S.py` 실행

## 의존성

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 파일별 설명

각 파일에 대한 자세한 설명은 아래 파일별 README를 참조하세요.
