import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 讀取 train_list.txt 和 test_list.txt
def load_data_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_path, file_id, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            data_list.append((file_path, file_id, start, end))
    return data_list

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

# def extract_features(file_path, fixed_length):
#     y, sr = librosa.load(file_path, sr=None)
#     if len(y) < fixed_length:
#         y = np.pad(y, (0, fixed_length - len(y)), mode='constant')
#     else:
#         y = y[:fixed_length]
    
#     y = zero_justification(y)
    
#     # 提取 MFCC（時間序列）
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     delta_mfcc = librosa.feature.delta(mfcc)
#     delta2_mfcc = librosa.feature.delta(mfcc, order=2)

#     # 組合特徵（不再取均值）
#     features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

#     return features.flatten()  # 攤平成 1D 陣列



def find_max_length(data_list):
    max_length = 0
    for file_path, _, _, _ in tqdm(data_list, desc="Finding max length"):
        y, sr = librosa.load(file_path, sr=None)
        if len(y) > max_length:
            max_length = len(y)
    return max_length

def calculate_accuracy(y_pred, y_true, tolerance, label_scaler):
    # 反標準化預測值和真實值
    y_pred = label_scaler.inverse_transform(y_pred.detach().numpy())
    y_true = label_scaler.inverse_transform(y_true.detach().numpy())

    # # 印出前十比 y_pred 和 y_true
    # print("y_pred: ", y_pred[:10])
    # print("y_true: ", y_true[:10])

    abs_error = np.abs(y_pred - y_true)  # 計算 |y_pred - y_true| 的絕對誤差
    correct_predictions = (abs_error <= tolerance).all(axis=1)  # 檢查每一筆數據是否在容許誤差範圍內
    accuracy = correct_predictions.mean()  # 計算準確率
    return accuracy

if __name__ == "__main__":
    # 讀取已經處理好的特徵和標籤
    train_features = np.load('../data/train_features.npy')
    test_features = np.load('../data/test_features.npy')
    train_labels = np.load('../data/train_labels.npy')
    test_labels = np.load('../data/test_labels.npy')
    feature_scaler = np.load('../data/feature_scaler.npy', allow_pickle=True).item()
    label_scaler = np.load('../data/label_scaler.npy', allow_pickle=True).item()

    # 轉換 numpy 陣列為 PyTorch tensor
    X_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(test_features, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # print("X_train shape: ", X_train.shape)
    # print("y_train shape: ", y_train.shape)
    # print(feature_scaler.inverse_transform(X_train[:5].detach().numpy()))
    # print(label_scaler.inverse_transform(y_train[:5].detach().numpy()))

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

            self.fc4 = nn.Linear(64, 2)  # 輸出層

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



    # 建立模型
    input_dim = X_train.shape[1]
    model = FNN(input_dim)

    # 設定損失函數和優化器
    criterion = nn.SmoothL1Loss()  # 使用 SmoothL1Loss 作為損失函數
    #criterion = nn.MSELoss()  # 使用 MSELoss 作為損失函數
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 將訓練資料轉換為 PyTorch tensor 並創建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 訓練模型
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 計算準確率
        tolerance = 1000  # 減小容許誤差範圍
        accuracy = calculate_accuracy(model(X_train), y_train, tolerance, label_scaler)

        # 印出前十次的訓練結果
        # if epoch < 20:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        #     print("預測值: ", label_scaler.inverse_transform(outputs[:5].detach().numpy()))
        #     print("真實值: ", label_scaler.inverse_transform(y_train[:5].detach().numpy()))

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    # 保存模型參數
    torch.save(model.state_dict(), '../model/model.pth')

    # 測試模型
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        test_accuracy = calculate_accuracy(predictions, y_test, tolerance, label_scaler)
        print(f"測試集 Loss: {test_loss.item():.4f}, Accuracy: {test_accuracy:.4f}")

    print("模型訓練與測試完成！")
