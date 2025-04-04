import os
import torch
import librosa
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 初始化 wandb 專案
wandb.init(project="music_emotion_deam", config={
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "sample_rate": 22050,
    "max_duration": 30,
    "n_mels": 128
})

# 全域參數
AUDIO_DIR = '/home/ouvic/ML/Music_emo/DEAM_audio/MEMD_audio'
CSV_PATH = '/home/ouvic/ML/Music_emo/DEAM_Annotations/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv'
SAMPLE_RATE = wandb.config.sample_rate
MAX_DURATION = wandb.config.max_duration  # seconds
MAX_LEN = SAMPLE_RATE * MAX_DURATION
N_MELS = wandb.config.n_mels

# ---------- Dataset ----------
class DEAMDataset(Dataset):
    def __init__(self, audio_dir, csv_path):
        self.annotations = pd.read_csv(csv_path)
        self.annotations.columns = self.annotations.columns.str.strip()
        self.audio_dir = audio_dir
        self.file_list = self.annotations['song_id'].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        song_id = self.file_list[idx]
        audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")

        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            # 回傳預設值 (全0 tensor, 中位標籤)
            return torch.zeros((1, N_MELS, MAX_LEN // 512)), torch.tensor([5.0, 5.0])
        
        y = librosa.util.fix_length(y, size=MAX_LEN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = torch.tensor(mel_db, dtype=torch.float32)
        mel_db = (mel_db - mel_db.mean()) / mel_db.std()

        # 讀取標籤
        val = self.annotations.loc[self.annotations['song_id'] == song_id, 'valence_mean'].values[0]
        aro = self.annotations.loc[self.annotations['song_id'] == song_id, 'arousal_mean'].values[0]
        target = torch.tensor([val, aro], dtype=torch.float32)

        return mel_db.unsqueeze(0), target  # shape: [1, N_MELS, T]

# ---------- 改進後的模型 ----------
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

# ---------- 新增的分類標籤函數 ----------
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

# ---------- Training ----------
def train():
    dataset = DEAMDataset(AUDIO_DIR, CSV_PATH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=wandb.config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.MSELoss()

    epochs = wandb.config.epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 計算訓練準確率
            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for p, t in zip(pred_np, y_np):
                pred_label = get_emotion_label(p[0], p[1])
                true_label = get_emotion_label(t[0], t[1])
                if pred_label == true_label:
                    train_correct += 1
                train_total += 1
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy, 
            "Epoch": epoch+1
        })
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

        # 驗證
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

                # 將預測與真實值轉換為情緒標籤進行比較
                pred_np = pred.cpu().numpy()
                y_np = y.cpu().numpy()
                for p, t in zip(pred_np, y_np):
                    pred_label = get_emotion_label(p[0], p[1])
                    true_label = get_emotion_label(t[0], t[1])
                    if pred_label == true_label:
                        correct += 1
                    total += 1

        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0
        wandb.log({"Val Loss": val_loss, "Val Accuracy": accuracy, "Epoch": epoch+1})
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), 'emotion_crnn_model.pth')
    print("模型已儲存為 emotion_crnn_model.pth")
    
    # 繪製混淆矩陣
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # 收集預測和真實標籤
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            for p, t in zip(pred_np, y_np):
                pred_label = get_emotion_label(p[0], p[1])
                true_label = get_emotion_label(t[0], t[1])
                y_pred.append(pred_label)
                y_true.append(true_label)

    # 建立混淆矩陣
    labels = ['Happy', 'Peaceful', 'Tense', 'Sad']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Emotion Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 儲存混淆矩陣圖片
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 上傳混淆矩陣到 wandb
    wandb.log({"Confusion Matrix": wandb.Image('confusion_matrix.png')})

if __name__ == '__main__':
    train()
