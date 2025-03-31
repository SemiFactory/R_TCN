import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import ConfusionMatrixDisplay

# 🔹 결과 데이터 경로 설정
result_dir = "/Users/kwonminseok/Desktop/restart_TCN/R_env/results"

# 🔹 저장된 예측 결과 로드
y_true = np.load(os.path.join(result_dir, "y_true.npy"))
y_pred = np.load(os.path.join(result_dir, "y_pred.npy"))
conf_matrix = np.load(os.path.join(result_dir, "conf_matrix.npy"))

# 🔹 클래스 정의 (0~3번 클래스)
class_labels = ["Class 0", "Class 1", "Class 2", "Class 3"]

# 🔹 1. 혼동 행렬 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))  # 결과 저장
plt.show()

# 🔹 2. 클래스별 정확도 계산 및 시각화
accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

plt.figure(figsize=(6, 4))
plt.bar(class_labels, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy")
plt.savefig(os.path.join(result_dir, "class_accuracy.png"))  # 결과 저장
plt.show()

print(f"✅ 시각화 완료! 저장 경로: {result_dir}")
