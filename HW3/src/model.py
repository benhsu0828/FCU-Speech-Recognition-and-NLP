import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
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