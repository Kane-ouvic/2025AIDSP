import os
import torch
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 設定參數
MODEL_PATH = '../pth/improved_emotion_model.pth'
SAMPLE_RATE = 22050
MAX_DURATION = 30
MAX_LEN = SAMPLE_RATE * MAX_DURATION
N_MELS = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_emotion_label(valence, arousal):
    # 根據 threshold=5 將連續值映射為情緒標籤
    if valence >= 5 and arousal >= 5:
        return "Happy"
    elif valence >= 5 and arousal < 5:
        return "Peaceful"
    elif valence < 5 and arousal >= 5:
        return "Tense"
    else:
        return "Sad"

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

def load_model(model_path):
    model = ImprovedEmotionCNN().to(device)  # 加上 .to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 支援 CPU/GPU 都可載入
    model.eval()
    return model



def predict_emotion(audio_path, model):
    # 載入音樂檔案
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=MAX_LEN)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = torch.tensor(mel_db, dtype=torch.float32)
    mel_db = (mel_db - mel_db.mean()) / mel_db.std()
    mel_db = mel_db.unsqueeze(0).unsqueeze(0)  # [1, 1, N_MELS, T]

    # 預測
    with torch.no_grad():
        pred = model(mel_db)
        valence, arousal = pred[0].numpy()
        emotion_label = get_emotion_label(valence, arousal)
    
    return emotion_label, valence, arousal

if __name__ == '__main__':
    # 載入模型
    model = load_model(MODEL_PATH)
    
    # 指定音樂檔案路徑
    audio_path = '../music/3.mp3'
    
    # 預測情緒
    emotion, valence, arousal = predict_emotion(audio_path, model)
    print(f"The predicted emotion for the audio file is: {emotion}")
    print(f"Valence: {valence}, Arousal: {arousal}")