import sys
import os
import subprocess
import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QLineEdit, QRadioButton, QGroupBox, QMessageBox, QSizePolicy, QStyleFactory
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QTimer, Qt

class FallDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("跌倒偵測 - 來源選擇")
        self.setGeometry(100, 100, 500, 300)
        
        self.detector_process = None  # 存放 subprocess.Popen 的物件
        self.sound_enabled = True  # 音效開關預設為開啟
        
        # 建立畫面元件
        self.initUI()
        
    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        
        # 影像來源群組
        src_group = QGroupBox("影像來源選擇")
        src_layout = QVBoxLayout()
        self.rb_webcam = QRadioButton("網路攝影機 (Webcam)")
        self.rb_webcam.setChecked(True)
        self.rb_rtsp = QRadioButton("RTSP 串流")
        src_layout.addWidget(self.rb_webcam)
        src_layout.addWidget(self.rb_rtsp)
        src_group.setLayout(src_layout)
        
        # RTSP URL 輸入
        rtsp_layout = QHBoxLayout()
        lbl_rtsp = QLabel("RTSP URL:")
        self.txt_rtsp = QLineEdit()
        self.txt_rtsp.setPlaceholderText("例如: rtsp://192.168.1.10:554/stream")
        self.txt_rtsp.setEnabled(False)
        rtsp_layout.addWidget(lbl_rtsp)
        rtsp_layout.addWidget(self.txt_rtsp)
        
        # 當選擇 RTSP 則啟用輸入框
        self.rb_rtsp.toggled.connect(lambda: self.txt_rtsp.setEnabled(self.rb_rtsp.isChecked()))
        
        # 控制按鈕
        self.btn_start_stop = QPushButton("開始偵測")
        self.btn_start_stop.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-start"))
        self.btn_start_stop.clicked.connect(self.toggle_detection)
        
        # 音效開關按鈕
        self.btn_toggle_sound = QPushButton("關閉音效")
        self.btn_toggle_sound.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_toggle_sound.setIcon(QIcon.fromTheme("audio-volume-high"))
        self.btn_toggle_sound.clicked.connect(self.toggle_sound)
        
        # 狀態顯示
        self.lbl_status = QLabel("尚未執行偵測")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setFont(QFont("Arial", 11))
        
        # 加入佈局
        self.main_layout.addWidget(src_group)
        self.main_layout.addLayout(rtsp_layout)
        self.main_layout.addWidget(self.btn_start_stop)
        self.main_layout.addWidget(self.btn_toggle_sound)
        self.main_layout.addWidget(self.lbl_status)
        
        self.main_layout.addStretch(1)
        
        self.apply_styles()
        self.add_history_display()  # 添加歷史記錄區域
        
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2D2D30; color: white; }
            QLabel { color: white; font-size: 11px; }
            QPushButton { 
                background-color: #0078D7; color: white; border: none;
                border-radius: 4px; padding: 8px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1C97EA; }
            QPushButton:pressed { background-color: #00559E; }
            QGroupBox { 
                border: 1px solid #3E3E42; border-radius: 6px; margin-top: 10px; 
                font-weight: bold; color: #00AAFF;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QLineEdit { 
                padding: 5px; border: 1px solid #555; border-radius: 3px; 
                background-color: #3A3A3D; color: white;
            }
            QRadioButton { color: white; spacing: 5px; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QMessageBox {
                background-color: #3E3E42; /* 深色背景 */
            }
            QMessageBox QLabel { /* QMessageBox 中的 QLabel */
                color: white; /* 白色文字 */
                background-color: transparent; /* 透明背景，以免覆蓋 QMessageBox 的背景 */
                font-size: 11px; /* 與其他 QLabel 一致或按需調整 */
            }
            QMessageBox QPushButton { /* QMessageBox 中的 QPushButton */
                background-color: #0078D7; 
                color: white;
                border: none;
                border-radius: 4px; 
                padding: 6px 10px; /* 彈出視窗按鈕的內邊距可稍小 */
                font-weight: bold;
            }
            QMessageBox QPushButton:hover { background-color: #1C97EA; }
            QMessageBox QPushButton:pressed { background-color: #00559E; }
        """)
        self.lbl_status.setFont(QFont("Arial", 10))
        self.lbl_status.setStyleSheet("color: #FFD700;")

    def add_history_display(self):
        # 添加歷史記錄顯示區域
        self.history_group = QGroupBox("歷史記錄")
        self.history_layout = QVBoxLayout()
        self.history_label = QLabel("尚無記錄")
        self.history_label.setWordWrap(True)
        self.history_layout.addWidget(self.history_label)
        self.history_group.setLayout(self.history_layout)
        self.main_layout.addWidget(self.history_group)

    def update_history(self, message):
        # 更新歷史記錄
        current_text = self.history_label.text()
        updated_text = f"{current_text}\n{message}" if current_text != "尚無記錄" else message
        self.history_label.setText(updated_text)

    def toggle_sound(self):
        # 添加音效開關功能
        self.sound_enabled = not self.sound_enabled
        status = "開啟" if self.sound_enabled else "關閉"
        self.lbl_status.setText(f"音效已{status}")
    
    def toggle_detection(self):
        # 如果目前偵測中則停止
        if self.detector_process is not None:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        # 根據選擇決定參數： webcam 用數字 0，RTSP 傳入 URL
        if self.rb_webcam.isChecked():
            video_param = "0"
        else:
            rtsp_url = self.txt_rtsp.text().strip()
            if not rtsp_url:
                QMessageBox.warning(self, "輸入錯誤", "請輸入有效的 RTSP URL")
                return
            video_param = rtsp_url
            
        # 組合命令 (假設偵測程式的入口在 main.py)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "main.py"), "--video", video_param]
        
        try:
            # 使用 preexec_fn=os.setsid 讓該程序使用獨立的 process group
            self.detector_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
            self.lbl_status.setText("偵測中...")
            self.btn_start_stop.setText("停止偵測")
            self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-stop"))
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法啟動跌倒偵測：{e}")
            self.detector_process = None

    def stop_detection(self):
        if self.detector_process is not None:
            try:
                # 發送 SIGTERM 給整個 process group
                os.killpg(os.getpgid(self.detector_process.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Error terminating process: {e}")
            self.detector_process.wait()
            self.detector_process = None
            self.lbl_status.setText("偵測已停止")
            self.btn_start_stop.setText("開始偵測")
            self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def closeEvent(self, event):
        # 關閉畫面前若有執行偵測則停止
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 嘗試使用 Fusion 風格
    available_styles = QStyleFactory.keys()
    if "Fusion" in available_styles:
        app.setStyle(QStyleFactory.create("Fusion"))
    
    win = FallDetectorGUI()
    win.show()
    sys.exit(app.exec_())