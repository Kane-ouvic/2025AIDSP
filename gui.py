import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal

from code import inference as inf
#### #################################################
import os
import torch
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#### #################################################
import sounddevice as sd
import time
class RecognizeThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, model, sample_rate, record_fn, predict_fn):
        super().__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.record_fn = record_fn
        self.predict_fn = predict_fn

    def run(self):
        for i in range(4):
            clip = self.record_fn()
            buffer = clip[:, 0]  # 取左聲道
            emotion, valence, arousal = self.predict_fn(buffer, self.model)
            msg = f"預測情緒: {emotion}\nValence: {valence:.2f}, Arousal: {arousal:.2f}"
            self.result_signal.emit(msg)
            time.sleep(2)

class EmotionRecognizerGUI(QWidget):
    MODEL_PATH = './pth/improved_emotion_model.pth'
    SAMPLE_RATE = inf.SAMPLE_RATE
    MAX_DURATION = inf.MAX_DURATION
    MAX_LEN = SAMPLE_RATE * MAX_DURATION
    N_MELS = inf.N_MELS
    device = inf.device
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音樂情緒辨識")
        self.setGeometry(100, 100, 300, 200)

        self.audio_file_path = None
        self.audio = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("請選擇音樂或錄音檔案", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.record_button = QPushButton("即興辨識", self)
        self.record_button.clicked.connect(self.realtime_recognize)
        layout.addWidget(self.record_button)

        self.open_file_button = QPushButton("開啟音樂檔案", self)
        self.open_file_button.clicked.connect(self.open_audio_file)
        layout.addWidget(self.open_file_button)

        self.recognize_button = QPushButton("辨識", self)
        self.recognize_button.clicked.connect(self.recognize_emotion)
        layout.addWidget(self.recognize_button)

        self.setLayout(layout)

    def record_clip(self):
        audio = sd.rec(int(15 * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=2, dtype='float32')
        sd.wait()
        return audio
    def realtime_recognize(self):
        self.label.setText("🎙️ 即興辨識中...")
        model = inf.load_model(self.MODEL_PATH)
        self.rec_thread = RecognizeThread(
            model=model,
            sample_rate=self.SAMPLE_RATE,
            record_fn=self.record_clip,
            predict_fn=self.predict_emotion
        )
        self.rec_thread.result_signal.connect(self.update_label)
        self.rec_thread.start()
            
    
    def open_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇音樂檔案", "", "音訊檔案 (*.wav *.mp3 *.flac)"
        )
        if file_name:
            self.audio_file_path = file_name
            self.label.setText(f"已選擇：{file_name.split('/')[-1]}")
            self.audio, sr = librosa.load(self.audio_file_path, sr=self.SAMPLE_RATE)
            
    def recognize_emotion(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "錯誤", "請先開啟音樂檔案或錄音")
            return

        # TODO: 加入音樂情緒辨識的程式邏輯
        # self.audio_file_path 是選擇的音訊檔案路徑
        self.label.setText("辨識中")
        
        model = inf.load_model(self.MODEL_PATH)
        emotion, valence, arousal = self.predict_emotion(self.audio, model)
        msg = f"The predicted emotion for the audio file is: {emotion}" + '\n' + \
              f"Valence: {valence}, Arousal: {arousal}"
        self.label.setText("辨識完成")
        QMessageBox.information(self, "辨識完成", msg)
    
    def predict_emotion(self, audio, model):
        # 載入音樂檔案
        y, sr = audio, self.SAMPLE_RATE
        y = librosa.util.fix_length(y, size=self.MAX_LEN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = torch.tensor(mel_db, dtype=torch.float32)
        
        mel_db = (mel_db - mel_db.mean()) / mel_db.std()
        mel_db = mel_db.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, N_MELS, T]

        # 預測
        with torch.no_grad():
            pred = model(mel_db)
            # valence, arousal = pred[0].numpy()
            valence, arousal = pred[0].cpu().numpy()
            emotion_label = inf.get_emotion_label(valence, arousal)
        
        return emotion_label, valence, arousal

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionRecognizerGUI()
    window.show()
    sys.exit(app.exec_())
