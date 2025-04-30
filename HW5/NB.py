import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# === Load Preprocessed Data ===
data = np.load('./data/mfcc_data.npz')
X = data["features"]  # shape: (samples, n_mfcc)
y = data["labels"]  # shape: (samples,)
splits = data["splits"]  # shape: (samples,)

# === Encode Speaker Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 將語者標籤轉換為數字編碼

# === Split Training and Testing Data ===
train_mask = splits == "training_data"
X_train = X[train_mask]
y_train = y_encoded[train_mask]
X_test = X[~train_mask]
y_test = y_encoded[~train_mask]

# === Standardize Features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 對訓練資料進行標準化
X_test = scaler.transform(X_test)  # 使用相同的標準化參數對測試資料進行標準化

# === Train Naive Bayes Model ===
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = nb_model.predict(X_test)

# === Classification Report ===
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Overall Accuracy ===
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")