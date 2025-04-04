import os
import torch
import librosa
import numpy as np
from train import ImprovedEmotionCNN, get_emotion_label

# 設定參數
MODEL_PATH = 'improved_emotion_model.pth'
SAMPLE_RATE = 22050
MAX_DURATION = 30
MAX_LEN = SAMPLE_RATE * MAX_DURATION
N_MELS = 128

def load_model(model_path):
    model = ImprovedEmotionCNN()
    model.load_state_dict(torch.load(model_path))
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