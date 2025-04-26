import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import mediapipe as mp
import joblib
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
import sys

from net import Net
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
        
    def initUI(self):
        self.setWindowTitle('風格轉換即時演示')
        self.setGeometry(100, 100, 1200, 700)
        
        # 主佈局
        main_layout = QVBoxLayout()
        
        # 影像顯示區域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)
        
        # 控制按鈕區域
        control_layout = QHBoxLayout()
        
        # 開始/停止按鈕
        self.start_button = QPushButton('開始')
        self.start_button.clicked.connect(self.toggleCamera)
        control_layout.addWidget(self.start_button)
        
        # 功能開關區域
        toggle_layout = QHBoxLayout()
        
        # 人像分割開關
        self.segment_checkbox = QCheckBox('人像分割')
        self.segment_checkbox.setChecked(self.segment_human)
        self.segment_checkbox.stateChanged.connect(self.toggleSegment)
        toggle_layout.addWidget(self.segment_checkbox)
        
        # 手勢辨識開關
        self.gesture_checkbox = QCheckBox('手勢辨識')
        self.gesture_checkbox.setChecked(self.detect_gesture)
        self.gesture_checkbox.stateChanged.connect(self.toggleGesture)
        toggle_layout.addWidget(self.gesture_checkbox)
        
        # 人數計數開關
        self.person_checkbox = QCheckBox('人數計數')
        self.person_checkbox.setChecked(self.detect_person)
        self.person_checkbox.stateChanged.connect(self.togglePerson)
        toggle_layout.addWidget(self.person_checkbox)
        
        # 風格切換按鈕
        self.prev_style_button = QPushButton('上一個風格')
        self.prev_style_button.clicked.connect(self.prevStyle)
        toggle_layout.addWidget(self.prev_style_button)
        
        self.next_style_button = QPushButton('下一個風格')
        self.next_style_button.clicked.connect(self.nextStyle)
        toggle_layout.addWidget(self.next_style_button)
        
        # 狀態顯示區域
        status_layout = QHBoxLayout()
        self.segment_status = QLabel('人像分割: 關閉')
        self.gesture_status = QLabel('手勢辨識: 關閉')
        self.person_status = QLabel('人數計數: 關閉')
        status_layout.addWidget(self.segment_status)
        status_layout.addWidget(self.gesture_status)
        status_layout.addWidget(self.person_status)
        
        # 將所有佈局添加到主佈局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(toggle_layout)
        main_layout.addLayout(status_layout)
        
        # 設置主窗口的中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 初始化計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def initModels(self):
        # 初始化 mediapipe selfie segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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
        try:
            print("正在載入 YOLO 模型...")
            self.yolo_model = YOLO('./models/yolov8n.pt')
            print("YOLO 模型載入成功")
        except Exception as e:
            print(f"YOLO 模型載入失敗: {str(e)}")
            self.detect_person = False
            self.person_checkbox.setChecked(False)
            self.person_checkbox.setEnabled(False)

        # 初始化風格轉換模型
        self.style_model = Net(ngf=self.args.ngf)
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
        
    def toggleCamera(self):
        if not self.running:
            self.running = True
            self.start_button.setText('停止')
            self.timer.start(30)  # 約 33 FPS
        else:
            self.running = False
            self.start_button.setText('開始')
            self.timer.stop()
            
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
        
    def nextStyle(self):
        self.style_idx = (self.style_idx + 1) % self.style_loader.size()
        
    def updateStatusLabels(self):
        self.segment_status.setText(f'人像分割: {"開啟" if self.segment_human else "關閉"}')
        self.gesture_status.setText(f'手勢辨識: {"開啟" if self.detect_gesture else "關閉"}')
        self.person_status.setText(f'人數計數: {"開啟" if self.detect_person else "關閉"}')
        
    def update_frame(self):
        ret_val, img = self.cam.read()
        if not ret_val:
            return
            
        if self.mirror:
            img = cv2.flip(img, 1)
        cimg = img.copy()

        # 人數計數
        if self.detect_person:
            try:
                results = self.yolo_model(cimg, stream=True)
                person_count = 0
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if self.yolo_model.names[cls_id] == 'person':
                            person_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(cimg, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(cimg, f"People Count: {person_count}", (20, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"YOLO 推論失敗: {str(e)}")
                self.detect_person = False
                self.person_checkbox.setChecked(False)
                self.updateStatusLabels()

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
                    
                    # 根據手勢切換風格
                    if confidence > 0.8:
                        if label != self.last_gesture:  # 只有當手勢改變時才切換
                            if label == "Good":
                                self.style_idx = (self.style_idx - 1) % self.style_loader.size()
                                self.last_gesture = label
                            elif label == "Bad":
                                self.style_idx = (self.style_idx + 1) % self.style_loader.size()
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
        style_v = Variable(style_v.data)
        self.style_model.setTarget(style_v)

        img_tensor = torch.from_numpy(img_for_style).unsqueeze(0).float()
        if self.args.cuda:
            img_tensor = img_tensor.cuda()

        img_tensor = Variable(img_tensor)
        img_tensor = self.style_model(img_tensor)

        if self.args.cuda:
            simg = style_v.cpu().data[0].numpy()
            img_stylized = img_tensor.cpu().clamp(0, 255).data[0].numpy()
        else:
            simg = style_v.data.numpy()
            img_stylized = img_tensor.clamp(0, 255).data[0].numpy()
        simg = np.squeeze(simg)
        img_stylized = img_stylized.transpose(1, 2, 0).astype('uint8')
        simg = simg.transpose(1, 2, 0).astype('uint8')

        # 將風格化結果與原始影像根據人像遮罩合併
        img_original = cimg.copy()
        if self.segment_human:
            result = np.where(condition_3ch > 0, img_original, img_stylized)
        else:
            result = img_stylized

        # 顯示風格縮圖
        simg = cv2.resize(simg, (self.swidth, self.sheight), interpolation=cv2.INTER_CUBIC)
        result[0:self.sheight, 0:self.swidth, :] = simg
        
        # 合併原始影像和風格化結果
        display_img = np.concatenate((cimg, result), axis=1)
        
        # 轉換為 QImage 並顯示
        h, w, c = display_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
        
    def closeEvent(self, event):
        self.timer.stop()
        self.cam.release()
        event.accept()

def run_demo(args):
    app = QApplication(sys.argv)
    window = StyleTransferApp(args)
    window.show()
    sys.exit(app.exec_())

def main():
    # getting things ready
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    # run demo
    run_demo(args)

if __name__ == '__main__':
    main()
