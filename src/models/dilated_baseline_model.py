import torch
import torch.nn as nn

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2*dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + x

class ATACSeqCNN(nn.Module):
    def __init__(self, sequence_length, num_targets):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=19, padding=9),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
        )
        self.dilated_blocks = nn.Sequential(
            DilatedBlock(256, 1),
            DilatedBlock(256, 2),
            DilatedBlock(256, 4),
            DilatedBlock(256, 8)
        )
        self.conv_tail = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        fc_input_dim = 256 * (sequence_length // (2**5))
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_targets),
            nn.Softplus()
        )

    def forward(self, x, reverse=False):
        if reverse:
            x = reverse_complement_tensor(x)
        x = self.stem(x)
        x = self.dilated_blocks(x)
        x = self.conv_tail(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
