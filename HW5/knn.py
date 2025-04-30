import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === Load Preprocessed Data ===
data = np.load('./data/mfcc_data.npz')
X = data["features"]  # shape: (samples, n_mfcc)
y = data["labels"]
splits = data["splits"]

# === 編碼語者標籤 ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 根據 split 分割訓練和測試集 ===
train_mask = splits == "training_data"
X_train = X[train_mask]
y_train = y_encoded[train_mask]
X_test = X[~train_mask]
y_test = y_encoded[~train_mask]

# === 建立與評估 KNN 模型 ===
# 測試不同的 K 值，k 從 1 到 20
best_k = 1
best_accuracy = 0.0
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
    print(f"Accuracy for k={k}: {accuracy:.4f}")

# === Step 6: Report ===
# 選用最好的 k 來計算總體準確率
print(f"Best k: {best_k}")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 使用 zero_division=0 避免警告
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# 計算總體準確率
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")