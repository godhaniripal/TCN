import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, in_ch=94, n_classes=3, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=8, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x, mask=None):
        # x: (B, F, T), mask: (B, T) with 1/0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if mask is not None:
            m = mask.unsqueeze(1)  # (B,1,T)
            x = (x * m).sum(-1) / (m.sum(-1) + 1e-6)  # (B,C)
        else:
            x = x.mean(-1)
        x = self.dropout(x)
        return self.head(x)