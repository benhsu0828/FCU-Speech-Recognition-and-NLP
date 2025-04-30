import os
import pandas as pd
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# === Config ===
CSV_PATH = './data/metadata.csv'
N_MFCC = 13  # 可調整的特徵數
SAMPLE_RATE = 16000

# === Step 1: Read Metadata ===
df = pd.read_csv(CSV_PATH)

# === Step 2: Extract MFCC Features ===
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # 平均壓縮成固定長度特徵向量
    return mfcc_mean

print('Extracting features...')
features = []
labels = []
splits = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    feat = extract_features(row['path'])
    features.append(feat)
    labels.append(row['speaker'])
    splits.append(row['split'])

features = np.array(features)
labels = np.array(labels)
splits = np.array(splits)

# === Step 3: Encode speaker labels ===
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# === Step 4: Split into train/eval/test ===
X_train = features[splits == 'training_data']
y_train = encoded_labels[splits == 'training_data']

X_eval = features[splits == 'eval_data']
y_eval = encoded_labels[splits == 'eval_data']

X_test = features[splits == 'testing_data']
y_test = encoded_labels[splits == 'testing_data']

# === Step 5: Grid Search K ===
print('Searching best k...')
best_k = 1
best_acc = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_eval, y_eval)
    print(f'k={k}, Eval Accuracy={acc:.4f}')
    if acc > best_acc:
        best_k = k
        best_acc = acc

# === Step 6: Retrain with best_k and test ===
print(f'Best k = {best_k}, training final model...')
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

print('Evaluating on test set...')
y_pred = final_knn.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 檢查類別分佈
unique, counts = np.unique(labels[splits == 'testing_data'], return_counts=True)
print("Test set class distribution:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")
