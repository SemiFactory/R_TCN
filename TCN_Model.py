import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import os

# 데이터 로드
npy_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/train_data/npy_files"
X_train = np.load(os.path.join(npy_dir, "X.npy"))
y_train = np.load(os.path.join(npy_dir, "y.npy"))

# 다중 클래스 분류를 위한 원-핫 인코딩 적용
num_classes = len(np.unique(y_train))  # 0, 1, 2, 3의 4개 클래스
y_train = to_categorical(y_train, num_classes=num_classes)

# TCN 모델 정의
def build_tcn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 첫 번째 TCN 블록
    x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', dilation_rate=1)(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    
    # 두 번째 TCN 블록
    x = layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu', dilation_rate=4)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu', dilation_rate=8)(x)
    x = layers.BatchNormalization()(x)
    
    # 글로벌 평균 풀링 및 출력층
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# 모델 생성
model = build_tcn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32
)

# 모델 저장
model_save_path = "/Users/kwonminseok/Desktop/restart_TCN/R_env/models/tcn_model.h5"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"모델 저장 완료: {model_save_path}")
