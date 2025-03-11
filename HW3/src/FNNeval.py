import torch
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# 定義 FNN 模型
class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Dropout 1

        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Dropout 2

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)  # Dropout 3

        self.fc4 = nn.Linear(64, 1)  # 輸出層

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)  # Apply dropout

        x = self.fc4(x)  # 輸出層不使用 Dropout
        return x

# 加載模型參數
input_dim = 13  # 假設輸入特徵維度為 13
model = FNN(input_dim)
model.load_state_dict(torch.load('../model/model.pth'))
model.eval()

# 加載標準化器
feature_scaler = np.load('../data/feature_scaler.npy', allow_pickle=True).item()

def zero_justification(raw_data):
    x = np.arange(len(raw_data))  # 確保 x 的長度與 raw_data 相同
    coefs = np.polyfit(x, raw_data, 3)
    coefs_point = np.polyval(coefs, x)
    return raw_data - coefs_point

def extract_features(file_path, frame_length=512, hop_length=192, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  # 讀取音檔
    y = zero_justification(y)  # 應用 Zero Justification
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)  # 取 n_mfcc 維 MFCC
    mfcc = mfcc.T  # 轉置，使每行代表一個 frame
    return mfcc, sr

# 針對特定資料進行測試
def test_model(file_path, frame_length=512, hop_length=192, n_mfcc=13):
    features, sr = extract_features(file_path, frame_length, hop_length, n_mfcc)
    features = feature_scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(features)
        predictions = torch.sigmoid(predictions).numpy()  # 使用 sigmoid 函數將輸出轉換為概率
        predictions = (predictions >= 0.5).astype(int).flatten()  # 將概率轉換為 0 或 1，並轉換為 1D 數組
        return predictions, sr

# 從文件名中提取真實值
def extract_true_values(file_path):
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    start = int(parts[1])
    end = int(parts[2].split('.')[0])
    return np.array([start, end])

# 找到預測為 1 的區間的開始和結束位置
def find_predicted_ranges(predictions):
    ranges = []
    start = None
    for i, pred in enumerate(predictions):
        if pred == 1 and start is None:
            start = i
        elif pred == 0 and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(predictions) - 1))
    return ranges

# 視覺化預測結果和真實結果
def visualize_results(file_path, predictions, true_values, frame_length=512, hop_length=192):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    
    # 計算每個 frame 的時間範圍
    frame_times = np.arange(len(predictions)) * hop_length / sr
    
    # 找到預測為 1 的區間的開始和結束位置
    predicted_ranges = find_predicted_ranges(predictions)
    
    # 畫出真實的起氣點和結束點
    plt.vlines(true_values / sr, ymin=y.min(), ymax=y.max(), color='blue', linestyle='--', label='True Values')
    
    # 畫出預測為 1 的區間的開始和結束位置
    for start, end in predicted_ranges:
        plt.vlines([frame_times[start], frame_times[end]], ymin=y.min(), ymax=y.max(), color='red', linestyle='--', label='Predicted Values')
    

    # 打印真實的起氣點和結束點
    print(f"True Values (start, end): {true_values / sr}")
    
    # 打印預測的起氣點和結束點
    predicted_times = [(frame_times[start], frame_times[end]) for start, end in predicted_ranges]
    print(f"Predicted Values (start, end): {predicted_times}")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform with True and Predicted Values')
    plt.legend()
    plt.show()
    
    

# 測試特定資料
test_file_path = "../data/wavefiles-all/960003_Jens/0b_3791_15371.wav"
predictions, sr = test_model(test_file_path)

# 從文件名中提取真實值
true_values = extract_true_values(test_file_path)

# 視覺化結果
visualize_results(test_file_path, predictions, true_values)