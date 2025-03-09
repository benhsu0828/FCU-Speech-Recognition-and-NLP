import torch
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import os

# 定義 FNN 模型
class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)  # 預測起氣點和結束點

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 加載模型參數
input_dim = 13  # 假設輸入特徵維度為 13
model = FNN(input_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 加載標準化器
feature_scaler = torch.load('feature_scaler.pkl')
label_scaler = torch.load('label_scaler.pkl')

def zero_justification(raw_data):
    x = np.arange(len(raw_data))  # 確保 x 的長度與 raw_data 相同
    coefs = np.polyfit(x, raw_data, 3)
    coefs_point = np.polyval(coefs, x)
    return raw_data - coefs_point

def extract_features(file_path, fixed_length):
    y, sr = librosa.load(file_path, sr=None)  # 讀取音檔
    if len(y) < fixed_length:
        y = np.pad(y, (0, fixed_length - len(y)), mode='constant')  # 填充音頻信號到固定長度
    else:
        y = y[:fixed_length]  # 截斷音頻信號到固定長度
    y = zero_justification(y)  # 應用 Zero Justification
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 取 13 維 MFCC
    return np.mean(mfcc, axis=1)  # 計算 MFCC 平均值

# 針對特定資料進行測試
def test_model(file_path, fixed_length):
    # 檢查文件路徑是否存在嵌入的空字符
    if '\x00' in file_path:
        raise ValueError("文件路徑中存在嵌入的空字符")

    # 檢查文件是否存在
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    features = extract_features(file_path, fixed_length)
    features = feature_scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(features)
        prediction = label_scaler.inverse_transform(prediction.numpy())
        print("預測值: ", prediction)

# 測試特定資料
test_file_path = 'E:/FUC-Speech-Recognition-and-NLP/HW3/wavefiles-all/M1305359/0a_10463_17664.wav'  # 替換為你想要測試的音頻文件路徑
fixed_length = 32000  # 根據你的固定長度設置
test_model(test_file_path, fixed_length)