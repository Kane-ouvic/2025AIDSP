import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ImprovedEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(N_MELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x: [B, 1, N_MELS, T]，先 squeeze channel 1
        x = x.squeeze(1)  # [B, N_MELS, T]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # [B, 64, 1]
        x = x.squeeze(-1)        # [B, 64]
        return self.fc(x)
    


class EmotionCRNN(nn.Module):
    def __init__(self, n_mels=128, rnn_hidden_size=128, rnn_layers=2, dropout=0.3):
        super(EmotionCRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 時間/頻率維度皆縮小

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # shape: [B, 128, n_mels/8, time/8]
        )

        # RNN 輸入維度計算：取 flatten 後的頻率維度
        self.rnn_input_size = (n_mels // 8) * 128

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 輸出 Valence 和 Arousal
        )

    def forward(self, x):
        # x: [B, 1, N_MELS, T]
        x = self.cnn(x)  # [B, 128, N_MELS//8, T//8]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T//8, C, F]
        x = x.view(b, t, -1)  # [B, T//8, C*F] -> RNN input

        rnn_out, _ = self.rnn(x)  # [B, T', 2*hidden]
        out = rnn_out[:, -1, :]   # 取最後一個時間步的輸出

        return self.fc(out)       # [B, 2]