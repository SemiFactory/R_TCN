# y 차원 검정 코드 


import numpy as np

# y_test 로드
y_test = np.load('/Users/kwonminseok/Desktop/restart_TCN/R_env/train_data/npy_files/y.npy')

# 차원 확인 
print("y_test shape:", y_test.shape)

# y_test가 1차원인지 확인
if len(y_test.shape) == 1:
    print("⚠️ Warning: y_test가 1차원입니다. 원-핫 인코딩이 아니라 정수 레이블일 가능성이 높습니다.")
else:
    print("✅ y_test는 원-핫 인코딩된 2차원 배열입니다.")
