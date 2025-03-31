import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix

# 🔹 경로 설정
model_path = "/Users/kwonminseok/Desktop/restart_TCN/R_env/models/tcn_model.h5"
v_npy_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/vaild_data/v_npy_files"
result_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/results"

# 🔹 저장된 모델 불러오기
print(f"모델 로드 중: {model_path}")
model = tf.keras.models.load_model(model_path)

# 🔹 검증 데이터 로드
X_test = np.load(os.path.join(v_npy_dir, "X.npy"))
y_test = np.load(os.path.join(v_npy_dir, "y_test.npy"))

# 🔹 예측 수행
y_pred_prob = model.predict(X_test)  # 확률값 예측
y_pred = np.argmax(y_pred_prob, axis=1)  # 가장 높은 확률의 클래스 선택
y_true = np.argmax(y_test, axis=1)  # 원-핫 인코딩을 정수 레이블로 변환

# 🔹 평가 결과 출력
print("📌 분류 리포트:")
print(classification_report(y_true, y_pred))

print("📌 혼동 행렬:")
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

# 🔹 결과 저장 (시각화를 위한 데이터 저장)
os.makedirs(result_dir, exist_ok=True)
np.save(os.path.join(result_dir, "y_true.npy"), y_true)
np.save(os.path.join(result_dir, "y_pred.npy"), y_pred)
np.save(os.path.join(result_dir, "conf_matrix.npy"), conf_matrix)

print(f"✅ 예측 결과 저장 완료! 저장 경로: {result_dir}")
