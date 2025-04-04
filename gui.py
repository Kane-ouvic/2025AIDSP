import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt


class EmotionRecognizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音樂情緒辨識")
        self.setGeometry(100, 100, 300, 200)

        self.audio_file_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("請選擇音樂或錄音檔案", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.record_button = QPushButton("錄音", self)
        self.record_button.clicked.connect(self.record_audio)
        layout.addWidget(self.record_button)

        self.open_file_button = QPushButton("開啟音樂檔案", self)
        self.open_file_button.clicked.connect(self.open_audio_file)
        layout.addWidget(self.open_file_button)

        self.recognize_button = QPushButton("辨識", self)
        self.recognize_button.clicked.connect(self.recognize_emotion)
        layout.addWidget(self.recognize_button)

        self.setLayout(layout)

    def record_audio(self):
        # TODO: 加入錄音功能
        QMessageBox.information(self, "錄音", "錄音功能尚未實作。")

    def open_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇音樂檔案", "", "音訊檔案 (*.wav *.mp3 *.flac)"
        )
        if file_name:
            self.audio_file_path = file_name
            self.label.setText(f"已選擇：{file_name.split('/')[-1]}")

    def recognize_emotion(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "錯誤", "請先開啟音樂檔案或錄音")
            return

        # TODO: 加入音樂情緒辨識的程式邏輯
        # self.audio_file_path 是選擇的音訊檔案路徑
        QMessageBox.information(self, "辨識中", "這裡會顯示辨識邏輯的結果。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionRecognizerGUI()
    window.show()
    sys.exit(app.exec_())
