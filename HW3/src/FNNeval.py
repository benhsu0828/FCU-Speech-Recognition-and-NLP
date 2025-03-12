import torch
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

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

# 檢測是否有可用的 CUDA 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載模型參數
input_dim = 13  # 假設輸入特徵維度為 13
model = FNN(input_dim).to(device)
model.load_state_dict(torch.load('../model/model.pth', map_location=device))
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
    features = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(features)
        predictions = torch.sigmoid(predictions).cpu().numpy()  # 使用 sigmoid 函數將輸出轉換為概率
        predictions = (predictions >= 0.5).astype(int).flatten()  # 將概率轉換為 0 或 1，並轉換為 1D 數組
        return predictions, sr, hop_length

# 從文件名中提取真實值
def extract_true_values(file_path):
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    start = int(parts[1])
    end = int(parts[2].split('.')[0])
    return np.array([start, end])

# 找到預測為 1 的區間的開始和結束位置
def find_predicted_ranges(predictions, hop_length):
    ranges = []
    start = None
    for i, pred in enumerate(predictions):
        if pred == 1 and start is None:
            start = i * hop_length
        elif pred == 0 and start is not None:
            ranges.append((start, (i - 1) * hop_length))
            start = None
    if start is not None:
        ranges.append((start, (len(predictions) - 1) * hop_length))
    return ranges

# 計算分數
def calculate_score(predicted_ranges, true_values, sr, time_diff):
    if len(predicted_ranges) == 0:
        return 0.0  # 如果沒有預測到任何區間，得分為0
    predicted_start = predicted_ranges[0][0]
    #如果沒有predicted_end，則predicted_end = predicted_start
    predicted_end = predicted_ranges[-1][1] if len(predicted_ranges) > 1 else predicted_start
    true_start, true_end = true_values
    start_diff = np.abs(predicted_start - true_start) / sr
    end_diff = np.abs(predicted_end - true_end) / sr
    if start_diff < time_diff and end_diff < time_diff:
        return 1.0  # 成功預測出起氣點和結束點，得分為1
    elif start_diff < time_diff or end_diff < time_diff:
        return 0.5  # 只預測出其中一個，得分為0.5
    else:
        return 0.0  # 都沒預測中，得分為0

# 主函數
def main():
    wav_files = glob.glob('../data/wavefiles-all/**/*.wav', recursive=True)
    scores = []
    time_diff = 0.0625  # 設置時間差異閾值

    with open('results.txt', 'w') as f:
        for file_path in tqdm(wav_files, desc="Processing files"):
            predictions, sr, hop_length = test_model(file_path)
            true_values = extract_true_values(file_path)
            predicted_ranges = find_predicted_ranges(predictions, hop_length)
            score = calculate_score(predicted_ranges, true_values, sr, time_diff)
            scores.append(score)
            f.write(f"File: {file_path},sample rate{sr} ,True Values: {true_values}, Predicted Ranges: {predicted_ranges}, Score: {score}\n")

    final_score = np.mean(scores)
    print(f"Final Score: {final_score}")

if __name__ == "__main__":
    main()