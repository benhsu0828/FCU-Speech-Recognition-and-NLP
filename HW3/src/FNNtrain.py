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
from model import FNN  # 導入模型

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

def find_max_length(data_list):
    max_length = 0
    for file_path, _, _, _ in tqdm(data_list, desc="Finding max length"):
        y, sr = librosa.load(file_path, sr=None)
        if len(y) > max_length:
            max_length = len(y)
    return max_length

def calculate_accuracy(y_pred, y_true, threshold=0.5):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    y_pred = (y_pred >= threshold).astype(int)
    accuracy = (y_pred == y_true).mean()
    return accuracy

if __name__ == "__main__":
    # 檢查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 讀取已經處理好的特徵和標籤
    train_features = np.load('../data/train_features.npy')
    test_features = np.load('../data/test_features.npy')
    train_labels = np.load('../data/train_labels.npy')
    test_labels = np.load('../data/test_labels.npy')
    feature_scaler = np.load('../data/feature_scaler.npy', allow_pickle=True).item()

    # 轉換 numpy 陣列為 PyTorch tensor
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)

    # 建立模型
    input_dim = X_train.shape[1]
    model = FNN(input_dim).to(device)

    # 設定損失函數和優化器
    criterion = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 作為損失函數
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 將訓練資料轉換為 PyTorch tensor 並創建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 調整批次大小

    # 訓練模型
    num_epochs = 50
    early_stop_patience = 5  # 早停的耐心值
    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 計算準確率
        accuracy = calculate_accuracy(model(X_train), y_train)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

            # 打印幾筆測試資料和運算結果
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test[:5])
                test_predictions = torch.sigmoid(test_outputs).cpu().numpy()
                test_predictions = (test_predictions >= 0.5).astype(int)  # 將大於或等於0.5的值歸類為1，小於0.5的值歸類為0
                test_labels_sample = y_test[:5].cpu().numpy()
                print("Sample Test Predictions: ", test_predictions)
                print("Sample Test Labels: ", test_labels_sample)

        # 驗證模型
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

        # 早停機制
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '../model/model.pth')  # 保存最佳模型參數
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 測試模型
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        test_accuracy = calculate_accuracy(predictions, y_test)
        print(f"測試集 Loss: {test_loss.item():.4f}, Accuracy: {test_accuracy:.4f}")

    print("模型訓練與測試完成！")
