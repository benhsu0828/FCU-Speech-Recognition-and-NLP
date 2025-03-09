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

def extract_features(file_path, fixed_length, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  # 讀取音檔
    if len(y) < fixed_length:
        y = np.pad(y, (0, fixed_length - len(y)), mode='constant')  # 填充音頻信號到固定長度
    else:
        y = y[:fixed_length]  # 截斷音頻信號到固定長度
    y = zero_justification(y)  # 應用 Zero Justification
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # 取 n_mfcc 維 MFCC
    return np.mean(mfcc, axis=1)  # 計算 MFCC 平均值

def find_max_length(data_list):
    max_length = 0
    for file_path, _, _, _ in data_list:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) > max_length:
            max_length = len(y)
    return max_length

if __name__ == "__main__":
    # 設定主目錄
    main_directory = 'E:/FUC-Speech-Recognition-and-NLP/HW3/data/wavefiles-all'

    # 遞迴搜尋所有 wav 檔案
    wav_files = glob.glob(os.path.join(main_directory, '**', '*.wav'), recursive=True)

    # 正則表達式來解析檔名 (格式: ID_起氣點_結束點.wav)
    pattern = re.compile(r'([^_/]+)_(\d+)_(\d+)\.wav')

    # 存放資料的列表
    data_entries = []

    for file_path in wav_files:
        file_name = os.path.basename(file_path)  # 取得檔名
        match = pattern.match(file_name)
        
        if match:
            file_id, start, end = match.groups()
            start, end = int(start), int(end)  # 轉為數字
            data_entries.append((file_path, file_id, start, end))

    # 檢查結果
    print(f"找到 {len(data_entries)} 筆資料")
    print(data_entries[:5])  # 顯示前五筆

    # 將資料分為訓練集與測試集
    train_data, test_data = train_test_split(data_entries, test_size=0.2, random_state=42)

    # 找到最長的音檔長度
    max_length = max(find_max_length(train_data), find_max_length(test_data))
    print(f"最長的音檔長度: {max_length}")

    # 轉換所有音檔為特徵
    n_mfcc = 13  # 設定 MFCC 的維度
    train_features = np.array([extract_features(f[0], max_length, n_mfcc) for f in train_data])
    test_features = np.array([extract_features(f[0], max_length, n_mfcc) for f in test_data])

    # 標籤 (模型要預測的起氣點和結束點)
    train_labels = np.array([[f[2], f[3]] for f in train_data])
    test_labels = np.array([[f[2], f[3]] for f in test_data])

    # 標準化特徵和標籤
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    train_features = feature_scaler.fit_transform(train_features)
    test_features = feature_scaler.transform(test_features)

    train_labels = label_scaler.fit_transform(train_labels)
    test_labels = label_scaler.transform(test_labels)

    # 保存特徵和標籤到文件
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/train_features.npy', train_features)
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/test_features.npy', test_features)
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/train_labels.npy', train_labels)
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/test_labels.npy', test_labels)
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/feature_scaler.npy', feature_scaler)
    np.save('E:/FUC-Speech-Recognition-and-NLP/HW3/data/label_scaler.npy', label_scaler)

    # 將分割結果存成 txt 檔 (可選)
    with open('E:/FUC-Speech-Recognition-and-NLP/HW3/data/train_list.txt', 'w') as train_file:
        for entry in train_data:
            train_file.write(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]}\n")

    with open('E:/FUC-Speech-Recognition-and-NLP/HW3/data/test_list.txt', 'w') as test_file:
        for entry in test_data:
            test_file.write(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]}\n")

    print(f"處理完成，訓練集: {len(train_data)} 筆, 測試集: {len(test_data)} 筆")