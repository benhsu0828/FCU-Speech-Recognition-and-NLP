import os
import glob
import re
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def zero_justification(raw_data):
    x = np.arange(len(raw_data))  # 確保 x 的長度與 raw_data 相同
    coefs = np.polyfit(x, raw_data, 3)
    coefs_point = np.polyval(coefs, x)
    return raw_data - coefs_point

def extract_features_and_labels(file_path, frame_length=512, hop_length=192, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  # 讀取音檔
    y = zero_justification(y)  # 應用 Zero Justification
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)  # 取 n_mfcc 維 MFCC
    mfcc = mfcc.T  # 轉置，使每行代表一個frame


    # 解析檔名以獲取起氣點和結束點
    file_name = os.path.basename(file_path)
    pattern = re.compile(r'([^_/]+)_(\d+)_(\d+)\.wav')
    match = pattern.match(file_name)
    if match:
        _, start, end = match.groups()
        start, end = int(start), int(end)
    else:
        raise ValueError(f"檔名格式不正確: {file_name}")

    # 計算每個frame的標籤
    labels = np.zeros(mfcc.shape[0])
    for i in range(mfcc.shape[0]):
        frame_start = i * hop_length
        frame_end = frame_start + frame_length
        if frame_start >= start and frame_end <= end:
            labels[i] = 1
        else:
            labels[i] = 0

    return mfcc, labels

if __name__ == "__main__":
    # 設定主目錄
    main_directory = '../data/wavefiles-all'

    # 遞迴搜尋所有 wav 檔案
    wav_files = glob.glob(os.path.join(main_directory, '**', '*.wav'), recursive=True)

    # 存放資料的列表
    all_features = []
    all_labels = []

    for file_path in wav_files:
        features, labels = extract_features_and_labels(file_path)
        all_features.append(features)
        all_labels.append(labels)

    # 將所有特徵和標籤合併
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)

    # 將資料分為訓練集與測試集
    train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    # 標準化特徵
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features)
    test_features = feature_scaler.transform(test_features)

    # 保存特徵和標籤到文件
    np.save('../data/train_features.npy', train_features)
    np.save('../data/test_features.npy', test_features)
    np.save('../data/train_labels.npy', train_labels)
    np.save('../data/test_labels.npy', test_labels)
    np.save('../data/feature_scaler.npy', feature_scaler)

    print(f"處理完成，訓練集: {len(train_labels)} 筆, 測試集: {len(test_labels)} 筆")