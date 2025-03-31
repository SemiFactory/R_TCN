# 즉, **훈련 데이터(`y_train`)에는 원-핫 인코딩이 적용되었지만, 검증 데이터(`y_test`)에는 적용되지 않아서 발생한 오류로 인해 오류를 해결하기위한 코드

import numpy as np
from tensorflow.keras.utils import to_categorical
import os

# 기존 검증 데이터 로드
npy_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/vaild_data/v_npy_files"
y_test = np.load(os.path.join(npy_dir, "y.npy"))

# 클래스 개수 확인
num_classes = len(np.unique(y_test))  # 예: 4개 클래스 (0, 1, 2, 3)

# 원-핫 인코딩 적용
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# 변환된 데이터 저장 (덮어쓰기)
np.save(os.path.join(npy_dir, "y_test.npy"), y_test_one_hot)
print("✅ y_test.npy에 원-핫 인코딩 적용 완료! 다시 실행해보세요.")
