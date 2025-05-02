import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast

import mediapipe as mp
import joblib
# from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox, QDialog, QFrame, QGroupBox, QSlider
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
import sys

from net import Net
from net_pruned import Net_pruned
from option import Options
import utils
from utils import StyleLoader

def compute_distances(landmarks):
    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    distances = []
    reference_distance = np.linalg.norm(
        np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]) -
        np.array([landmarks.landmark[9].x, landmarks.landmark[9].y])
    )

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance/reference_distance)  # Normalize the distance

    return distances

class VideoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('即時影像')
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #2D2D30; color: white;")
        
        layout = QVBoxLayout()
        
        # 創建標題標籤
        title_label = QLabel('風格轉換即時預覽')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 14, QFont.Bold))
        title_label.setStyleSheet("color: #00AAFF; margin: 10px;")
        layout.addWidget(title_label)
        
        # 創建一個框架來包含影像標籤
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: #1E1E1E; border: 2px solid #3E3E42; border-radius: 8px;")
        frame_layout = QVBoxLayout(frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        frame_layout.addWidget(self.image_label)
        
        layout.addWidget(frame)
        
        # 添加說明文字
        info_label = QLabel('左側為原始影像，右側為風格化後的影像')
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #CCCCCC; font-style: italic;")
        layout.addWidget(info_label)
        
        self.setLayout(layout)

class StreamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('串流影像')
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #2D2D30; color: white;")
        
        layout = QVBoxLayout()
        
        # 創建標題標籤
        title_label = QLabel('風格轉換即時預覽')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 14, QFont.Bold))
        title_label.setStyleSheet("color: #00AAFF; margin: 10px;")
        layout.addWidget(title_label)
        
        # 創建一個框架來包含影像標籤
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: #1E1E1E; border: 2px solid #3E3E42; border-radius: 8px;")
        frame_layout = QVBoxLayout(frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        frame_layout.addWidget(self.image_label)
        
        layout.addWidget(frame)
        
        # 添加說明文字
        info_label = QLabel('左側為原始影像，右側為風格化後的影像')
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #CCCCCC; font-style: italic;")
        layout.addWidget(info_label)
        
        # 添加連接狀態標籤
        self.connection_status = QLabel('連接狀態: 未連接')
        self.connection_status.setAlignment(Qt.AlignCenter)
        self.connection_status.setStyleSheet("color: #FF5555; font-weight: bold;")
        layout.addWidget(self.connection_status)
        
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.stream = None
        self.running = False
        
    def start_stream(self):
        if not self.running:
            self.stream = cv2.VideoCapture("rtmp://140.116.56.6:1935/live")
            if not self.stream.isOpened():
                self.connection_status.setText('連接狀態: 連接失敗')
                self.connection_status.setStyleSheet("color: #FF5555; font-weight: bold;")
                return False
            self.running = True
            self.timer.start(30)
            self.connection_status.setText('連接狀態: 已連接')
            self.connection_status.setStyleSheet("color: #55FF55; font-weight: bold;")
            return True
        return False
        
    def stop_stream(self):
        if self.running:
            self.timer.stop()
            if self.stream:
                self.stream.release()
            self.running = False
            self.connection_status.setText('連接狀態: 已斷開')
            self.connection_status.setStyleSheet("color: #FFAA55; font-weight: bold;")
            
    def update_stream(self):
        if self.stream and self.running:
            ret, frame = self.stream.read()
            if not ret:
                self.stop_stream()
                self.connection_status.setText('連接狀態: 串流中斷')
                self.connection_status.setStyleSheet("color: #FF5555; font-weight: bold;")
                return
                
            # 獲取父窗口的風格轉換相關屬性和方法
            parent = self.parent()
            if parent:
                img = frame
                cimg = img.copy()
                
                # 人數計數
                # if parent.detect_person:
                #     try:
                #         results = parent.yolo_model(cimg, stream=True)
                #         person_count = 0
                #         for result in results:
                #             boxes = result.boxes
                #             for box in boxes:
                #                 cls_id = int(box.cls[0])
                #                 if parent.yolo_model.names[cls_id] == 'person':
                #                     person_count += 1
                #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                #                     cv2.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #                     cv2.putText(cimg, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                #         cv2.putText(cimg, f"People Count: {person_count}", (20, parent.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     except Exception as e:
                #         print(f"YOLO 推論失敗: {str(e)}")
                
                # 手勢辨識
                if parent.detect_gesture:
                    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    hand_results = parent.hands.process(rgb_frame)
                    
                    if hand_results.multi_hand_landmarks:
                        for index, landmarks in enumerate(hand_results.multi_hand_landmarks):
                            # 分辨左右手
                            hand_label = "Right" if hand_results.multi_handedness[index].classification[0].label == "Left" else "Left"
                            
                            distances = compute_distances(landmarks)
                            distances = parent.scaler.transform([distances])
                            
                            prediction = parent.clf.predict(distances)
                            confidence = np.max(parent.clf.predict_proba(distances))
                            
                            label = parent.labels[prediction[0]]
                            display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"
                            
                            cv2.putText(cimg, display_text, (10, 30 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            # 顯示手部關鍵點
                            mp.solutions.drawing_utils.draw_landmarks(cimg, landmarks, parent.mp_hands.HAND_CONNECTIONS)
                            
                            # 根據手勢切換功能
                            if confidence > 0.8:
                                if label != parent.last_gesture:  # 只有當手勢改變時才切換
                                    if label == "Good":
                                        parent.style_idx = (parent.style_idx - 1) % parent.style_loader.size()
                                        parent.style_slider.setValue(parent.style_idx)
                                        parent.style_label.setText(f'風格 #{parent.style_idx+1}')
                                        parent.last_gesture = label
                                    elif label == "Bad":
                                        parent.style_idx = (parent.style_idx + 1) % parent.style_loader.size()
                                        parent.style_slider.setValue(parent.style_idx)
                                        parent.style_label.setText(f'風格 #{parent.style_idx+1}')
                                        parent.last_gesture = label
                                    elif label == "Cool":
                                        # 切換人像遮罩
                                        parent.segment_human = not parent.segment_human
                                        parent.segment_checkbox.setChecked(parent.segment_human)
                                        parent.updateStatusLabels()
                                        parent.last_gesture = label
                                    elif label == "Ya":
                                        # 切換計數功能
                                        parent.detect_person = not parent.detect_person
                                        parent.person_checkbox.setChecked(parent.detect_person)
                                        parent.updateStatusLabels()
                                        parent.last_gesture = label
                    else:
                        parent.last_gesture = None  # 當沒有檢測到手時重置上一次的手勢
                
                # 人像分割
                if parent.segment_human:
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = parent.selfie_segmentation.process(image_rgb)
                    mask = results.segmentation_mask
                    condition = mask > 0.5
                    condition = condition.astype(np.uint8) * 255
                    condition_3ch = cv2.merge([condition, condition, condition])
                else:
                    condition_3ch = np.zeros_like(img)
                
                # 風格轉換
                img_for_style = np.array(img).transpose(2, 0, 1)
                style_v = parent.style_loader.get(parent.style_idx)
                
                img_tensor = torch.from_numpy(img_for_style).unsqueeze(0).float()
                if parent.args.cuda:
                    img_tensor = img_tensor.cuda()
                    style_v = style_v.cuda()
                
                with torch.no_grad():
                    parent.style_model.setTarget(style_v)
                    img_stylized = parent.style_model(img_tensor).clamp(0, 255)[0]
                
                # 處理風格圖像
                if parent.args.cuda:
                    simg = style_v[0].cpu().contiguous()
                    img_stylized = img_stylized.cpu()
                    # 清空 CUDA 緩存以優化記憶體使用
                    torch.cuda.empty_cache()
                else:
                    simg = style_v[0].contiguous()
                
                # 轉換為 numpy 格式以便 OpenCV 處理
                simg = simg.permute(1, 2, 0).byte().numpy()
                img_stylized = img_stylized.permute(1, 2, 0).byte().numpy()
                
                # 將風格化結果與原始影像根據人像遮罩合併
                img_original = cimg.copy()
                if parent.segment_human:
                    result = np.where(condition_3ch > 0, img_original, img_stylized)
                else:
                    result = img_stylized
                
                # 顯示風格縮圖在右側影像的左上角
                simg_resized = cv2.resize(simg, (parent.swidth, parent.sheight), interpolation=cv2.INTER_CUBIC)
                
                # 在右側影像的左上角添加風格縮圖
                result[0:parent.sheight, 0:parent.swidth, :] = simg_resized
                
                # 確保 result 是 numpy 數組並且數據類型正確
                result = np.ascontiguousarray(result, dtype=np.uint8)
                
                # 添加風格信息
                cv2.putText(result, f"Style {parent.style_idx+1}", (10, parent.sheight + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 合併原始影像和風格化結果
                display_img = np.concatenate((cimg, result), axis=1)
                
                # 轉換為 QImage 並顯示在彈跳視窗中
                h, w, c = display_img.shape
                bytes_per_line = 3 * w
                q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            else:
                h, w, c = frame.shape
                bytes_per_line = 3 * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            
    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

class StyleTransferApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mirror = True
        self.segment_human = False
        self.detect_gesture = False
        self.detect_person = False
        self.running = False
        self.style_idx = 0
        self.last_gesture = None
        
        self.initUI()
        self.initModels()
        
        # 創建視頻顯示對話框
        self.video_dialog = VideoDialog(self)
        
        # 創建串流顯示對話框
        self.stream_dialog = StreamDialog(self)
        
    def initUI(self):
        self.setWindowTitle('風格轉換即時演示')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #00559E;
            }
            QCheckBox {
                color: white;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QGroupBox {
                border: 1px solid #3E3E42;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #00AAFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)
        
        # 主佈局
        main_layout = QVBoxLayout()
        
        # 標題
        title_label = QLabel('風格轉換即時演示系統')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setStyleSheet("color: #00AAFF; margin: 10px 0 20px 0;")
        main_layout.addWidget(title_label)
        
        # 控制按鈕區域
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 開始/停止按鈕
        self.start_button = QPushButton('開始')
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.toggleCamera)
        control_layout.addWidget(self.start_button)
        
        # 串流按鈕
        self.stream_button = QPushButton('開啟串流')
        self.stream_button.setIcon(QIcon.fromTheme("network-wireless"))
        self.stream_button.setMinimumHeight(40)
        self.stream_button.clicked.connect(self.toggleStream)
        control_layout.addWidget(self.stream_button)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 功能開關區域
        features_group = QGroupBox("功能設定")
        toggle_layout = QVBoxLayout()
        
        # 人像分割開關
        segment_layout = QHBoxLayout()
        self.segment_checkbox = QCheckBox('人像分割')
        self.segment_checkbox.setChecked(self.segment_human)
        self.segment_checkbox.stateChanged.connect(self.toggleSegment)
        segment_layout.addWidget(self.segment_checkbox)
        self.segment_status = QLabel('狀態: 關閉')
        self.segment_status.setStyleSheet("color: #AAAAAA;")
        segment_layout.addWidget(self.segment_status)
        toggle_layout.addLayout(segment_layout)
        
        # 手勢辨識開關
        gesture_layout = QHBoxLayout()
        self.gesture_checkbox = QCheckBox('手勢辨識')
        self.gesture_checkbox.setChecked(self.detect_gesture)
        self.gesture_checkbox.stateChanged.connect(self.toggleGesture)
        gesture_layout.addWidget(self.gesture_checkbox)
        self.gesture_status = QLabel('狀態: 關閉')
        self.gesture_status.setStyleSheet("color: #AAAAAA;")
        gesture_layout.addWidget(self.gesture_status)
        toggle_layout.addLayout(gesture_layout)
        
        # 人數計數開關
        person_layout = QHBoxLayout()
        self.person_checkbox = QCheckBox('人數計數')
        self.person_checkbox.setChecked(self.detect_person)
        self.person_checkbox.stateChanged.connect(self.togglePerson)
        person_layout.addWidget(self.person_checkbox)
        self.person_status = QLabel('狀態: 關閉')
        self.person_status.setStyleSheet("color: #AAAAAA;")
        person_layout.addWidget(self.person_status)
        toggle_layout.addLayout(person_layout)
        
        features_group.setLayout(toggle_layout)
        main_layout.addWidget(features_group)
        
        # 風格選擇區域
        style_group = QGroupBox("風格選擇")
        style_layout = QVBoxLayout()
        
        style_buttons_layout = QHBoxLayout()
        # 風格切換按鈕
        self.prev_style_button = QPushButton('上一個風格')
        self.prev_style_button.setIcon(QIcon.fromTheme("go-previous"))
        self.prev_style_button.clicked.connect(self.prevStyle)
        style_buttons_layout.addWidget(self.prev_style_button)
        
        self.style_label = QLabel('風格 #1')
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_label.setFont(QFont('Arial', 12))
        style_buttons_layout.addWidget(self.style_label)
        
        self.next_style_button = QPushButton('下一個風格')
        self.next_style_button.setIcon(QIcon.fromTheme("go-next"))
        self.next_style_button.clicked.connect(self.nextStyle)
        style_buttons_layout.addWidget(self.next_style_button)
        
        style_layout.addLayout(style_buttons_layout)
        
        # 添加風格滑塊
        self.style_slider = QSlider(Qt.Horizontal)
        self.style_slider.setMinimum(0)
        self.style_slider.setMaximum(20)  # 假設有21種風格
        self.style_slider.setValue(0)
        self.style_slider.setTickPosition(QSlider.TicksBelow)
        self.style_slider.setTickInterval(1)
        self.style_slider.valueChanged.connect(self.onStyleSliderChanged)
        style_layout.addWidget(self.style_slider)
        
        style_group.setLayout(style_layout)
        main_layout.addWidget(style_group)
        
        # 手勢提示區域
        gesture_info_group = QGroupBox("手勢控制提示")
        gesture_info_layout = QVBoxLayout()
        
        gesture_info = QLabel("• 豎起大拇指(Good): 切換到上一個風格\n"
                             "• 拇指向下(Bad): 切換到下一個風格\n"
                             "• 手勢Cool: 切換人像遮罩功能\n"
                             "• 手勢Ya: 切換人數計數功能")
        gesture_info.setStyleSheet("color: #CCCCCC;")
        gesture_info_layout.addWidget(gesture_info)
        
        gesture_info_group.setLayout(gesture_info_layout)
        main_layout.addWidget(gesture_info_group)
        
        # 設置主窗口的中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 初始化計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def onStyleSliderChanged(self, value):
        self.style_idx = value
        self.style_label.setText(f'風格 #{value+1}')
        
    def initModels(self):
        # 初始化 mediapipe selfie segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

        # 初始化手勢辨識
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        # 載入手勢模型
        model_filename = "models/svm_model.pkl"
        self.clf = joblib.load(model_filename)
        scaler_filename = "models/scaler.pkl"
        self.scaler = joblib.load(scaler_filename)
        
        # 載入標籤
        label_file = "models/labels.txt"
        with open(label_file, 'r') as f:
            self.labels = f.readlines()
        self.labels = [label.strip() for label in self.labels]

        # 初始化 YOLO 模型
        # try:
        #     print("正在載入 YOLO 模型...")
        #     self.yolo_model = YOLO('./models/yolov8n.pt')
        #     print("YOLO 模型載入成功")
        # except Exception as e:
        #     print(f"YOLO 模型載入失敗: {str(e)}")
        #     self.detect_person = False
        #     self.person_checkbox.setChecked(False)
        #     self.person_checkbox.setEnabled(False)

        # 清空 CUDA 緩存以優化記憶體使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 初始化風格轉換模型
        self.style_model = Net_pruned(ngf=self.args.ngf)
        model_dict = torch.load(self.args.model)
        model_dict_clone = model_dict.copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        self.style_model.load_state_dict(model_dict, False)
        self.style_model.eval()
        if self.args.cuda:
            self.style_loader = StyleLoader(self.args.style_folder, self.args.style_size)
            self.style_model.cuda()
        else:
            self.style_loader = StyleLoader(self.args.style_folder, self.args.style_size, False)

        # 設置攝影機
        self.height = self.args.demo_size
        self.width = int(4.0/3*self.args.demo_size)
        self.swidth = int(self.width/4)
        self.sheight = int(self.height/4)
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, self.width)
        self.cam.set(4, self.height)
        
        # 更新風格滑塊的最大值
        self.style_slider.setMaximum(self.style_loader.size() - 1)
        
    def toggleCamera(self):
        if not self.running:
            self.running = True
            self.start_button.setText('停止')
            self.start_button.setStyleSheet("background-color: #D32F2F;")
            self.timer.start(30)  # 約 33 FPS
            self.video_dialog.show()
        else:
            self.running = False
            self.start_button.setText('開始')
            self.start_button.setStyleSheet("background-color: #0078D7;")
            self.timer.stop()
            self.video_dialog.hide()
            
    def toggleStream(self):
        if not self.stream_dialog.isVisible():
            if self.stream_dialog.start_stream():
                self.stream_button.setText('關閉串流')
                self.stream_button.setStyleSheet("background-color: #D32F2F;")
                self.stream_dialog.show()
            else:
                print("無法開啟串流")
        else:
            self.stream_dialog.stop_stream()
            self.stream_dialog.hide()
            self.stream_button.setText('開啟串流')
            self.stream_button.setStyleSheet("background-color: #0078D7;")
            
    def toggleSegment(self, state):
        self.segment_human = state == Qt.Checked
        self.updateStatusLabels()
        
    def toggleGesture(self, state):
        self.detect_gesture = state == Qt.Checked
        self.updateStatusLabels()
        
    def togglePerson(self, state):
        self.detect_person = state == Qt.Checked
        self.updateStatusLabels()
        
    def prevStyle(self):
        self.style_idx = (self.style_idx - 1) % self.style_loader.size()
        self.style_slider.setValue(self.style_idx)
        self.style_label.setText(f'風格 #{self.style_idx+1}')
        
    def nextStyle(self):
        self.style_idx = (self.style_idx + 1) % self.style_loader.size()
        self.style_slider.setValue(self.style_idx)
        self.style_label.setText(f'風格 #{self.style_idx+1}')
        
    def updateStatusLabels(self):
        self.segment_status.setText(f'狀態: {"開啟" if self.segment_human else "關閉"}')
        self.segment_status.setStyleSheet(f"color: {'#4CAF50' if self.segment_human else '#AAAAAA'};")
        
        self.gesture_status.setText(f'狀態: {"開啟" if self.detect_gesture else "關閉"}')
        self.gesture_status.setStyleSheet(f"color: {'#4CAF50' if self.detect_gesture else '#AAAAAA'};")
        
        self.person_status.setText(f'狀態: {"開啟" if self.detect_person else "關閉"}')
        self.person_status.setStyleSheet(f"color: {'#4CAF50' if self.detect_person else '#AAAAAA'};")
        
    def update_frame(self):
        ret_val, img = self.cam.read()
        if not ret_val:
            return
            
        if self.mirror:
            img = cv2.flip(img, 1)
        cimg = img.copy()

        # 人數計數
        # if self.detect_person:
        #     try:
        #         results = self.yolo_model(cimg, stream=True)
        #         person_count = 0
        #         for result in results:
        #             boxes = result.boxes
        #             for box in boxes:
        #                 cls_id = int(box.cls[0])
        #                 if self.yolo_model.names[cls_id] == 'person':
        #                     person_count += 1
        #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
        #                     cv2.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #                     cv2.putText(cimg, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        #         cv2.putText(cimg, f"People Count: {person_count}", (20, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     except Exception as e:
        #         print(f"YOLO 推論失敗: {str(e)}")
        #         self.detect_person = False
        #         self.person_checkbox.setChecked(False)
        #         self.updateStatusLabels()

        # 手勢辨識
        if self.detect_gesture:
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for index, landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # 分辨左右手
                    hand_label = "Right" if hand_results.multi_handedness[index].classification[0].label == "Left" else "Left"
                    
                    distances = compute_distances(landmarks)
                    distances = self.scaler.transform([distances])
                    
                    prediction = self.clf.predict(distances)
                    confidence = np.max(self.clf.predict_proba(distances))
                    
                    label = self.labels[prediction[0]]
                    display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"
                    
                    cv2.putText(cimg, display_text, (10, 30 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # 顯示手部關鍵點
                    mp.solutions.drawing_utils.draw_landmarks(cimg, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 根據手勢切換功能
                    if confidence > 0.8:
                        if label != self.last_gesture:  # 只有當手勢改變時才切換
                            if label == "Good":
                                self.style_idx = (self.style_idx - 1) % self.style_loader.size()
                                self.style_slider.setValue(self.style_idx)
                                self.style_label.setText(f'風格 #{self.style_idx+1}')
                                self.last_gesture = label
                            elif label == "Bad":
                                self.style_idx = (self.style_idx + 1) % self.style_loader.size()
                                self.style_slider.setValue(self.style_idx)
                                self.style_label.setText(f'風格 #{self.style_idx+1}')
                                self.last_gesture = label
                            elif label == "Cool":
                                # 切換人像遮罩
                                self.segment_human = not self.segment_human
                                self.segment_checkbox.setChecked(self.segment_human)
                                self.updateStatusLabels()
                                self.last_gesture = label
                            elif label == "Ya":
                                # 切換計數功能
                                self.detect_person = not self.detect_person
                                self.person_checkbox.setChecked(self.detect_person)
                                self.updateStatusLabels()
                                self.last_gesture = label
            else:
                self.last_gesture = None  # 當沒有檢測到手時重置上一次的手勢

        # 人像分割
        if self.segment_human:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(image_rgb)
            mask = results.segmentation_mask
            condition = mask > 0.5
            condition = condition.astype(np.uint8) * 255
            condition_3ch = cv2.merge([condition, condition, condition])
        else:
            condition_3ch = np.zeros_like(img)

        # 風格轉換
        img_for_style = np.array(img).transpose(2, 0, 1)
        style_v = self.style_loader.get(self.style_idx)
        
        img_tensor = torch.from_numpy(img_for_style).unsqueeze(0).float()
        if self.args.cuda:
            img_tensor = img_tensor.cuda()
            style_v = style_v.cuda()

        with torch.no_grad():
            self.style_model.setTarget(style_v)
            img_stylized = self.style_model(img_tensor).clamp(0, 255)[0]
            
        # 處理風格圖像
        if self.args.cuda:
            simg = style_v[0].cpu().contiguous()
            img_stylized = img_stylized.cpu()
            # 清空 CUDA 緩存以優化記憶體使用
            torch.cuda.empty_cache()
        else:
            simg = style_v[0].contiguous()
            
        # 轉換為 numpy 格式以便 OpenCV 處理
        simg = simg.permute(1, 2, 0).byte().numpy()
        img_stylized = img_stylized.permute(1, 2, 0).byte().numpy()
        


        # 將風格化結果與原始影像根據人像遮罩合併
        img_original = cimg.copy()
        if self.segment_human:
            result = np.where(condition_3ch > 0, img_original, img_stylized)
        else:
            result = img_stylized
        # 顯示風格縮圖在右側影像的左上角
        simg_resized = cv2.resize(simg, (self.swidth, self.sheight), interpolation=cv2.INTER_CUBIC)
        
        # 在右側影像的左上角添加風格縮圖
        result[0:self.sheight, 0:self.swidth, :] = simg_resized
        
        # 確保 result 是 numpy 數組並且數據類型正確
        result = np.ascontiguousarray(result, dtype=np.uint8)
        
        # 添加風格信息
        cv2.putText(result, f"Style {self.style_idx+1}", (10, self.sheight + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 合併原始影像和風格化結果
        display_img = np.concatenate((cimg, result), axis=1)
        
        # 添加分隔線
        # 轉換為 QImage 並顯示在彈跳視窗中
        h, w, c = display_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_dialog.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_dialog.image_label.width(), self.video_dialog.image_label.height(), Qt.KeepAspectRatio))
    def closeEvent(self, event):
        self.timer.stop()
        self.cam.release()
        self.video_dialog.close()
        self.stream_dialog.close()
        # 清空 CUDA 緩存以釋放記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        event.accept()

def run_demo(args):
    app = QApplication(sys.argv)
    window = StyleTransferApp(args)
    window.show()
    sys.exit(app.exec_())

def main():
    print("main start")
    args = Options().parse()
    print(vars(args))
    if args.subcommand is None:
        print("subcommand is None")
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        print("cuda not available")
        raise ValueError("ERROR: cuda is not available, try running on CPU")
    # 清空 CUDA 緩存以優化記憶體使用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("run_demo start")
    run_demo(args)

if __name__ == '__main__':
    main()
    # test