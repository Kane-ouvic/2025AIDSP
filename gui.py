import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal

from code import inference as inf
from code import models as mdls
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
import pyaudio
import librosa.display
import matplotlib.pyplot as plt

class RecognizeThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, model, sample_rate, record_fn, predict_fn):
        super().__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.record_fn = record_fn
        self.predict_fn = predict_fn

    def run(self):
        buf_len = 10
        clip = self.record_fn()
        length = len(clip)
        buffer = np.zeros(length *buf_len)
        for i in range(1, buf_len):
            clip = self.record_fn()
            print(f"{i} clip stored...")
            buffer[i * length: length * (i+1)] = clip[:, 0]  # 取左聲道
        for i in range(15):
            emotion, valence, arousal = self.predict_fn(buffer, self.model)
            msg = f"預測情緒: {emotion}\nValence: {valence:.2f}, Arousal: {arousal:.2f}"
            self.result_signal.emit(msg)
            clip = self.record_fn()
            buffer[0:-len(clip)-1] = buffer[len(clip):-1]
            buffer[-len(clip)-1:-1] = clip[:, 0]
            time.sleep(1)
        msg = f"預測情緒: {emotion}\nValence: {valence:.2f}, Arousal: {arousal:.2f}\nEND"
        self.result_signal.emit(msg)
class EmotionRecognizerGUI(QWidget):
    MODEL_PATH = './pth/emotion_crnn_model.pth'
    SAMPLE_RATE = inf.SAMPLE_RATE
    MAX_DURATION = inf.MAX_DURATION
    MAX_LEN = SAMPLE_RATE * MAX_DURATION
    N_MELS = inf.N_MELS
    device = inf.device
    model = mdls.EmotionCRNN().to(device)  # 加上 .to(device)

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

        # 添加 Discord Bot 控制按鈕
        self.discord_bot_button = QPushButton("啟動 Discord Bot", self)
        self.discord_bot_button.clicked.connect(self.toggle_discord_bot)
        layout.addWidget(self.discord_bot_button)

        self.discord_play_button = QPushButton("在 Discord 播放", self)
        self.discord_play_button.clicked.connect(self.play_in_discord)
        self.discord_play_button.setEnabled(False)  # 初始時禁用
        layout.addWidget(self.discord_play_button)

        self.setLayout(layout)

    def toggle_discord_bot(self):
        try:
            from bot import start_bot, stop_bot
            
            if self.discord_bot_button.text() == "啟動 Discord Bot":
                # 啟動 bot
                TOKEN = ''  # 請替換成您的 Discord Bot Token
                if start_bot(TOKEN):
                    self.discord_bot_button.setText("關閉 Discord Bot")
                    self.discord_play_button.setEnabled(True)
                    QMessageBox.information(self, "成功", "Discord Bot 已啟動")
                else:
                    QMessageBox.warning(self, "警告", "Discord Bot 已在運行中")
            else:
                # 關閉 bot
                if stop_bot():
                    self.discord_bot_button.setText("啟動 Discord Bot")
                    self.discord_play_button.setEnabled(False)
                    QMessageBox.information(self, "成功", "Discord Bot 已關閉")
                else:
                    QMessageBox.warning(self, "警告", "Discord Bot 未在運行")
        except ImportError as e:
            QMessageBox.critical(self, "錯誤", f"無法載入 Discord Bot 模組: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"操作失敗: {str(e)}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # 支援 CPU/GPU 都可載入
        self.model.eval()
        return self.model

    def record_clip(self):
        audio = sd.rec(int(3 * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=2, dtype='float32')
        sd.wait()
        return audio
    def record_clip_pyaudio(self, duration=3, sample_rate=16000, channels=1):
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("🎙️ Recording...")
        frames = []
        for _ in range(0, int(sample_rate / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.float32))
        print("🛑 Done.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio = np.hstack(frames)
        if channels == 2:
            audio = audio.reshape(-1, 2)
        else:
            audio = audio.reshape(-1, 1)

        return audio

    def realtime_recognize(self):
        self.label.setText("🎙️ 即興辨識中...")
        model = self.load_model(self.MODEL_PATH)
        self.rec_thread = RecognizeThread(
            model=model,
            sample_rate=self.SAMPLE_RATE,
            #record_fn=self.record_clip,
            record_fn = lambda: self.record_clip_pyaudio(duration=3, sample_rate=self.SAMPLE_RATE, channels=2),
            predict_fn=self.predict_emotion
        )
        self.rec_thread.result_signal.connect(self.update_label)
        self.rec_thread.start()
            
    def update_label(self, text):
        self.label.setText(text)
        
    def open_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇音樂檔案", "", "音訊檔案 (*.wav *.mp3 *.flac)"
        )
        if file_name:
            self.audio_file_path = file_name
            self.label.setText(f"已選擇：{file_name.split('/')[-1]}")
            self.audio, sr = librosa.load(self.audio_file_path, sr=self.SAMPLE_RATE)
            self.plot_MFCC(self.audio, sr)
    def recognize_emotion(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "錯誤", "請先開啟音樂檔案或錄音")
            return

        # TODO: 加入音樂情緒辨識的程式邏輯
        # self.audio_file_path 是選擇的音訊檔案路徑
        self.label.setText("辨識中")
        
        model = self.load_model(self.MODEL_PATH)
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

    def plot_MFCC(self, audio, sr):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # 通常取前13維

        # 繪圖
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()

    def play_in_discord(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "錯誤", "請先開啟音樂檔案")
            return
            
        try:
            # 複製音樂檔案到 Discord bot 的音樂目錄
            import shutil
            import os
            import sys
            
            # 獲取當前文件的目錄
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 確保 music 目錄存在
            music_dir = os.path.join(current_dir, 'music')
            if not os.path.exists(music_dir):
                os.makedirs(music_dir)
                
            # 複製檔案
            file_name = os.path.basename(self.audio_file_path)
            target_path = os.path.join(music_dir, file_name)
            
            # 檢查源文件和目標文件是否相同
            if os.path.normpath(self.audio_file_path) != os.path.normpath(target_path):
                shutil.copy2(self.audio_file_path, target_path)
            
            # 設定自動控制參數
            try:
                from bot import setup_auto_control
                # 使用您的 Discord 頻道 ID
                channel_id = 851380042117283860  # 請替換成您的頻道ID
                setup_auto_control(channel_id, file_name)
                
                # 等待一下確保設置生效
                import time
                time.sleep(2)
                
                QMessageBox.information(self, "成功", f"已設定在 Discord 播放 {file_name}\n請檢查 Discord 頻道是否有音樂播放")
            except ImportError as e:
                QMessageBox.critical(self, "錯誤", f"無法載入 Discord Bot 模組: {str(e)}\n請確保已安裝 discord.py 並設置了 Bot Token")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"設定 Discord Bot 失敗: {str(e)}")
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"播放設定失敗: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionRecognizerGUI()
    window.show()
    sys.exit(app.exec_())
