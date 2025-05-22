import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QLineEdit, QRadioButton, QGroupBox, QMessageBox, QSizePolicy,
                             QStyleFactory)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize

# --- Helper functions from demo.py (or similar) ---
def euclidean_distance(kps1_flat, kps2_flat):
    """計算兩組展平關鍵點之間的歐氏距離。"""
    return np.linalg.norm(kps1_flat - kps2_flat)

def cosine_similarity_calc(kps1_flat, kps2_flat):
    """計算兩組展平關鍵點之間的餘弦相似度。"""
    dot_product = np.dot(kps1_flat, kps2_flat)
    norm_kps1 = np.linalg.norm(kps1_flat)
    norm_kps2 = np.linalg.norm(kps2_flat)
    if norm_kps1 == 0 or norm_kps2 == 0:
        return 0.0  # 若任一向量範數為零，則相似度為0
    similarity = dot_product / (norm_kps1 * norm_kps2)
    return similarity  # 範圍從 -1 到 1

class PoseComparator:
    def __init__(self, similarity_method='cosine'):
        self.similarity_method = similarity_method.lower()

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,        
            model_complexity=1,             
            smooth_landmarks=True,          
            min_detection_confidence=0.5,   
            min_tracking_confidence=0.5     
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 使用字典來儲存影格，以減少記憶體使用
        self.reference_frames = {}
        self.reference_keypoints_timeline = {}
        self.reference_results_timeline = {}
        self.current_ref_frame_index = 0
        self.ref_video_path = None
        self.total_frames = 0

        self.total_similarity_score = 0.0
        self.num_comparisons = 0
        self.is_reference_loaded = False
        
        # 記憶體管理參數
        self.window_size = 300  # 滑動視窗大小
        self.cleanup_threshold = 200  # 清理閾值

    def _cleanup_old_frames(self, current_index):
        """清理超出滑動視窗範圍的舊影格"""
        start_idx = max(0, current_index - self.window_size)
        end_idx = current_index + self.window_size
        
        # 清理超出範圍的影格
        keys_to_remove = []
        for idx in self.reference_frames.keys():
            if idx < start_idx or idx > end_idx:
                keys_to_remove.append(idx)
        
        for idx in keys_to_remove:
            self.reference_frames.pop(idx, None)
            self.reference_keypoints_timeline.pop(idx, None)
            self.reference_results_timeline.pop(idx, None)

    def _extract_keypoints_from_results(self, results):
        """從 MediaPipe results 物件中提取關鍵點。"""
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(landmarks)  # 形狀: (33, 4)
        return np.zeros((33, 4))  # 若未偵測到姿勢，返回零矩陣

    def _get_relevant_keypoints_for_similarity(self, keypoints_matrix):
        """
        選擇用於相似度計算的關鍵點座標 (例如 x, y) 並展平。
        目前使用 x, y 座標。
        """
        if np.all(keypoints_matrix == 0):
            return np.zeros(33 * 2) # 33 個關鍵點 * 2 個座標 (x, y)
        return keypoints_matrix[:, :2].flatten()  # 使用 x, y 座標並展平為一維陣列

    def load_reference_video(self, ref_video_path):
        self.ref_video_path = ref_video_path
        self.reference_frames.clear()
        self.reference_keypoints_timeline.clear()
        self.reference_results_timeline.clear()
        self.is_reference_loaded = False
        self.total_frames = 0

        cap_ref = cv2.VideoCapture(self.ref_video_path)
        if not cap_ref.isOpened():
            print(f"錯誤：無法開啟參考影片: {self.ref_video_path}")
            return False
        
        print("正在處理參考影片...")
        frame_idx = 0
        while cap_ref.isOpened():
            ret, frame = cap_ref.read()
            if not ret:
                break
            
            # 只儲存當前視窗範圍內的影格
            if frame_idx < self.window_size:
                frame_copy = np.copy(frame)
                self.reference_frames[frame_idx] = frame_copy
                
                image_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = self.pose_detector.process(image_rgb)
                image_rgb.flags.writeable = True
                
                self.reference_results_timeline[frame_idx] = results
                keypoints = self._extract_keypoints_from_results(results)
                self.reference_keypoints_timeline[frame_idx] = keypoints
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"已處理 {frame_idx} 個參考影格...")
                import gc
                gc.collect()

        cap_ref.release()
        self.total_frames = frame_idx
        
        if not self.reference_frames:
            print("警告：參考影片為空或無法讀取。")
            return False
        
        print(f"完成處理參考影片：共 {self.total_frames} 個影格。")
        self.is_reference_loaded = True
        self.reset_comparison()
        return True

    def _resize_frame(self, frame, target_height=None, target_width=None):
        """將影格大小調整至目標尺寸，同時保持寬高比（如果只提供一個維度）。"""
        h, w = frame.shape[:2]
        if h == 0 or w == 0: return frame # Avoid division by zero if frame is invalid

        if target_width and target_height: # Resize to specific dimensions
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        elif target_height: # Resize to target height, maintain aspect ratio
            if h == target_height: return frame
            scale = target_height / h
            return cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)
        elif target_width: # Resize to target width, maintain aspect ratio
            if w == target_width: return frame
            scale = target_width / w
            return cv2.resize(frame, (target_width, int(h*scale)), interpolation=cv2.INTER_AREA)
        return frame # No resize needed or specified

    def calculate_similarity_score(self, kps_user_flat, kps_ref_flat):
        """根據選擇的方法計算相似度分數。"""
        if np.all(kps_user_flat == 0) or np.all(kps_ref_flat == 0):
            return 0.0  # 若任一姿勢未偵測到，相似度為0

        if self.similarity_method == 'cosine':
            sim = cosine_similarity_calc(kps_user_flat, kps_ref_flat)
            return max(0, sim) # 餘弦相似度範圍為 [-1, 1]，標準化到 [0, 1]
        elif self.similarity_method == 'euclidean':
            dist = euclidean_distance(kps_user_flat, kps_ref_flat)
            # 將歐氏距離轉換為相似度分數 (0 到 1，越高越好)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"不支援的相似度計算方法: {self.similarity_method}")

    def process_user_frame(self, frame_user, target_display_h):
        """處理單一使用者影格，與當前參考影格比較，並返回處理後的影格及相似度。"""
        if not self.is_reference_loaded or self.current_ref_frame_index >= self.total_frames:
            blank_ref_w = frame_user.shape[1]
            blank_ref = np.zeros((target_display_h, blank_ref_w, 3), dtype=np.uint8)
            
            if target_display_h:
                frame_user_display = self._resize_frame(np.copy(frame_user), target_height=target_display_h)
            else:
                frame_user_display = np.copy(frame_user)
            return frame_user_display, blank_ref, 0.0, True

        try:
            # 確保當前影格在記憶體中
            if self.current_ref_frame_index not in self.reference_frames:
                # 重新讀取需要的影格
                cap_ref = cv2.VideoCapture(self.ref_video_path)
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, self.current_ref_frame_index)
                ret, frame = cap_ref.read()
                if ret:
                    frame_copy = np.copy(frame)
                    self.reference_frames[self.current_ref_frame_index] = frame_copy
                    
                    image_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = self.pose_detector.process(image_rgb)
                    image_rgb.flags.writeable = True
                    
                    self.reference_results_timeline[self.current_ref_frame_index] = results
                    keypoints = self._extract_keypoints_from_results(results)
                    self.reference_keypoints_timeline[self.current_ref_frame_index] = keypoints
                cap_ref.release()

            frame_ref_original = np.copy(self.reference_frames[self.current_ref_frame_index])
            kps_ref_matrix = self.reference_keypoints_timeline[self.current_ref_frame_index]

            frame_user_display = self._resize_frame(np.copy(frame_user), target_height=target_display_h)
            frame_ref_display = self._resize_frame(np.copy(frame_ref_original), target_height=target_display_h)
            
            image_user_rgb = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
            image_user_rgb.flags.writeable = False
            results_user = self.pose_detector.process(image_user_rgb)
            image_user_rgb.flags.writeable = True
            kps_user_matrix = self._extract_keypoints_from_results(results_user)

            kps_user_flat = self._get_relevant_keypoints_for_similarity(kps_user_matrix)
            kps_ref_flat = self._get_relevant_keypoints_for_similarity(kps_ref_matrix)

            current_similarity = 0.0
            if not (np.all(kps_user_flat == 0) or np.all(kps_ref_flat == 0)):
                current_similarity = self.calculate_similarity_score(kps_user_flat, kps_ref_flat)
                self.total_similarity_score += current_similarity
                self.num_comparisons += 1

            if results_user.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_user_display, results_user.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            ref_results_for_drawing = self.reference_results_timeline[self.current_ref_frame_index]
            if ref_results_for_drawing and ref_results_for_drawing.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_ref_display, ref_results_for_drawing.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2)
                )

            self.current_ref_frame_index += 1
            is_done = self.current_ref_frame_index >= self.total_frames

            # 清理舊影格
            if self.current_ref_frame_index % self.cleanup_threshold == 0:
                self._cleanup_old_frames(self.current_ref_frame_index)
                import gc
                gc.collect()

            return frame_user_display, frame_ref_display, current_similarity, is_done

        except Exception as e:
            print(f"處理影格時發生錯誤: {e}")
            return frame_user, np.zeros((target_display_h, frame_user.shape[1], 3), dtype=np.uint8), 0.0, True

    def get_reference_frame_count(self):
        return self.total_frames if self.is_reference_loaded else 0

    def get_current_frame_num(self):
        return min(self.current_ref_frame_index, self.total_frames) if self.is_reference_loaded else 0

    def reset_comparison(self):
        self.current_ref_frame_index = 0
        self.total_similarity_score = 0.0
        self.num_comparisons = 0
        # 重新載入初始影格
        self._cleanup_old_frames(0)

    def get_overall_similarity(self):
        if self.num_comparisons == 0:
            return 0.0
        return self.total_similarity_score / self.num_comparisons

    def get_evaluation_report_text(self, overall_similarity):
        """根據整體相似度分數產生評估報告文字。"""
        report = f"最終分數: {overall_similarity:.2f}\n"
        if overall_similarity > 0.85:
            report += "評估: 非常出色！您的動作與參考影片非常相似。"
        elif overall_similarity > 0.70:
            report += "評估: 良好！您的動作相當相似，但仍有進步空間。"
        elif overall_similarity > 0.50:
            report += "評估: 一般。存在明顯差異，請專注於更準確地匹配姿勢。"
        else:
            report += "評估: 需要改進。偵測到顯著差異，請仔細檢視參考動作。"
        return report
        
    def close(self):
        """清理所有資源"""
        if self.pose_detector:
            self.pose_detector.close()
        self.reference_frames.clear()
        self.reference_keypoints_timeline.clear()
        self.reference_results_timeline.clear()
        import gc
        gc.collect()


class DanceAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舞蹈姿態比較")
        self.setGeometry(50, 50, 1350, 750) 

        self.pose_comparator = PoseComparator(similarity_method='cosine') 
        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.selected_video_source = "webcam" 
        self.stream_url = ""
        
        self.target_display_height = 480 # 處理後的影像高度

        self.initUI()

    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget) 

        # --- 控制面板 (左側) ---
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_panel.setFixedWidth(320)

        # 參考影片群組
        ref_video_group = QGroupBox("參考影片")
        ref_video_layout = QVBoxLayout()
        self.btn_load_ref = QPushButton("載入參考影片")
        self.btn_load_ref.setIcon(QIcon.fromTheme("document-open", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-open-16.png"))) # Fallback icon
        self.btn_load_ref.clicked.connect(self.load_reference_video)
        ref_video_layout.addWidget(self.btn_load_ref)
        self.lbl_ref_video_path = QLabel("尚未載入參考影片")
        self.lbl_ref_video_path.setWordWrap(True)
        ref_video_layout.addWidget(self.lbl_ref_video_path)
        ref_video_group.setLayout(ref_video_layout)
        controls_layout.addWidget(ref_video_group)

        # 輸入來源群組
        input_source_group = QGroupBox("影像來源")
        input_source_layout = QVBoxLayout()
        self.rb_webcam = QRadioButton("網路攝影機")
        self.rb_webcam.setChecked(True)
        self.rb_webcam.toggled.connect(lambda: self.set_video_source("webcam"))
        input_source_layout.addWidget(self.rb_webcam)
        
        self.rb_stream = QRadioButton("串流")
        self.rb_stream.toggled.connect(lambda: self.set_video_source("stream"))
        input_source_layout.addWidget(self.rb_stream)
        
        self.txt_stream_url = QLineEdit()
        self.txt_stream_url.setPlaceholderText("輸入串流 URL (例如 RTMP, RTSP)")
        self.txt_stream_url.setEnabled(False) 
        input_source_layout.addWidget(self.txt_stream_url)
        input_source_group.setLayout(input_source_layout)
        controls_layout.addWidget(input_source_group)

        # 操作按鈕群組
        action_group = QGroupBox("控制")
        action_layout = QVBoxLayout()
        self.btn_start_stop = QPushButton("開始比較")
        self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-start", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-play-16.png")))
        self.btn_start_stop.clicked.connect(self.toggle_comparison)
        action_layout.addWidget(self.btn_start_stop)
        action_group.setLayout(action_layout)
        controls_layout.addWidget(action_group)

        # 狀態顯示群組
        status_group = QGroupBox("狀態")
        status_layout = QVBoxLayout()
        self.lbl_similarity_score = QLabel("即時相似度: N/A")
        status_layout.addWidget(self.lbl_similarity_score)
        self.lbl_frame_count = QLabel("影格: N/A")
        status_layout.addWidget(self.lbl_frame_count)
        self.lbl_overall_score = QLabel("整體評分: N/A")
        status_layout.addWidget(self.lbl_overall_score)
        status_group.setLayout(status_layout)
        controls_layout.addWidget(status_group)
        
        controls_layout.addStretch(1) 
        main_layout.addWidget(controls_panel)

        # --- 影像顯示區域 (右側) ---
        video_area_widget = QWidget()
        video_area_layout = QVBoxLayout(video_area_widget)

        video_display_layout = QHBoxLayout() 
        
        self.video_label_user = QLabel()
        self.video_label_user.setAlignment(Qt.AlignCenter)
        self.video_label_user.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.video_label_user.setMinimumSize(480, self.target_display_height) 
        self.video_label_user.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.video_label_ref = QLabel()
        self.video_label_ref.setAlignment(Qt.AlignCenter)
        self.video_label_ref.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.video_label_ref.setMinimumSize(480, self.target_display_height) 
        self.video_label_ref.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_display_layout.addWidget(self.video_label_user)
        video_display_layout.addWidget(self.video_label_ref)
        
        video_area_layout.addLayout(video_display_layout)

        title_layout = QHBoxLayout()
        lbl_user_title = QLabel("您的影像 (User)")
        lbl_user_title.setAlignment(Qt.AlignCenter)
        lbl_user_title.setFont(QFont("Arial", 12, QFont.Bold))
        title_layout.addWidget(lbl_user_title)

        lbl_ref_title = QLabel("參考影片 (Reference)")
        lbl_ref_title.setAlignment(Qt.AlignCenter)
        lbl_ref_title.setFont(QFont("Arial", 12, QFont.Bold))
        title_layout.addWidget(lbl_ref_title)
        video_area_layout.addLayout(title_layout)

        main_layout.addWidget(video_area_widget, 1) 

        self.apply_styles()


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
        self.lbl_similarity_score.setFont(QFont("Arial", 10))
        self.lbl_frame_count.setFont(QFont("Arial", 10))
        self.lbl_overall_score.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_overall_score.setStyleSheet("color: #FFD700;") 


    def load_reference_video(self):
        if self.is_running:
            QMessageBox.warning(self, "警告", "請先停止目前的比較，再載入新的參考影片。")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇參考影片檔案", "", 
                                                   "影片檔案 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*)", options=options)
        if file_path:
            self.lbl_ref_video_path.setText(f"載入中: {os.path.basename(file_path)}")
            QApplication.processEvents() 
            
            success = self.pose_comparator.load_reference_video(file_path)
            if success:
                self.lbl_ref_video_path.setText(f"已載入: {os.path.basename(file_path)}\n(共 {self.pose_comparator.get_reference_frame_count()} 幀)")
                self.lbl_overall_score.setText("整體評分: N/A") 
                if self.pose_comparator.reference_frames:
                    first_ref_frame = self.pose_comparator.reference_frames[0]
                    ref_results_for_drawing = self.pose_comparator.reference_results_timeline[0]
                    if ref_results_for_drawing and ref_results_for_drawing.pose_landmarks:
                        # Draw on a copy for display
                        display_copy = self.pose_comparator._resize_frame(first_ref_frame.copy(), target_height=self.target_display_height)
                        self.pose_comparator.mp_drawing.draw_landmarks(
                            display_copy, ref_results_for_drawing.pose_landmarks, self.pose_comparator.mp_pose.POSE_CONNECTIONS,
                            self.pose_comparator.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.pose_comparator.mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2)
                        )
                        self.display_image(display_copy, self.video_label_ref)
                    else: # No landmarks, just show resized frame
                        first_ref_frame_resized = self.pose_comparator._resize_frame(first_ref_frame, target_height=self.target_display_height)
                        self.display_image(first_ref_frame_resized, self.video_label_ref)

            else:
                self.lbl_ref_video_path.setText("載入失敗，請選擇有效的影片檔案。")
                QMessageBox.critical(self, "錯誤", f"無法載入或處理參考影片: {file_path}")


    def set_video_source(self, source):
        self.selected_video_source = source
        if source == "webcam":
            self.txt_stream_url.setEnabled(False)
        else: 
            self.txt_stream_url.setEnabled(True)

    def toggle_comparison(self):
        if not self.pose_comparator.is_reference_loaded:
            QMessageBox.warning(self, "提示", "請先載入參考影片。")
            return

        if self.is_running:
            self.stop_comparison()
        else:
            self.start_comparison()

    def start_comparison(self):
        source_ok = False
        if self.selected_video_source == "webcam":
            self.video_capture = cv2.VideoCapture(0) 
            if self.video_capture.isOpened():
                source_ok = True
            else:
                 QMessageBox.critical(self, "錯誤", "無法開啟網路攝影機。")
        else: 
            self.stream_url = self.txt_stream_url.text()
            if not self.stream_url:
                QMessageBox.warning(self, "提示", "請輸入串流 URL。")
                return
            self.video_capture = cv2.VideoCapture(self.stream_url)
            if self.video_capture.isOpened():
                source_ok = True
            else:
                QMessageBox.critical(self, "錯誤", f"無法開啟串流: {self.stream_url}")
        
        if source_ok:
            self.is_running = True
            self.btn_start_stop.setText("停止比較")
            self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-stop", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-stop-16.png")))
            self.pose_comparator.reset_comparison() 
            self.lbl_overall_score.setText("整體評分: 進行中...")
            self.timer.start(33) # Approx 30 FPS
            self.rb_webcam.setEnabled(False)
            self.rb_stream.setEnabled(False)
            self.txt_stream_url.setEnabled(False)
            self.btn_load_ref.setEnabled(False)


    def stop_comparison(self, finished_naturally=False):
        self.is_running = False
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.btn_start_stop.setText("開始比較")
        self.btn_start_stop.setIcon(QIcon.fromTheme("media-playback-start", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-play-16.png")))
        self.rb_webcam.setEnabled(True)
        self.rb_stream.setEnabled(True)
        self.txt_stream_url.setEnabled(self.rb_stream.isChecked())
        self.btn_load_ref.setEnabled(True)

        if self.pose_comparator.is_reference_loaded and self.pose_comparator.num_comparisons > 0:
            overall_sim = self.pose_comparator.get_overall_similarity()
            report_text = self.pose_comparator.get_evaluation_report_text(overall_sim)
            self.lbl_overall_score.setText(f"整體評分: {overall_sim:.2f}")
            if finished_naturally:
                QMessageBox.information(self, "比較完成", report_text)
        elif finished_naturally and self.pose_comparator.is_reference_loaded:
             self.lbl_overall_score.setText("整體評分: 無有效比較")
             QMessageBox.information(self, "比較完成", "沒有進行有效的姿態比較。")
        else: 
            self.lbl_similarity_score.setText("即時相似度: N/A")
            if not self.pose_comparator.is_reference_loaded:
                 self.lbl_overall_score.setText("整體評分: N/A")


    def update_frame(self):
        if not self.video_capture or not self.video_capture.isOpened():
            self.stop_comparison()
            QMessageBox.warning(self, "錯誤", "影像來源已中斷。")
            return

        ret, user_frame_original = self.video_capture.read()
        if not ret:
            print("警告：無法從使用者來源讀取影格。")
            if self.selected_video_source == "stream": 
                 self.stop_comparison(finished_naturally=True)
            return

        user_frame_flipped = cv2.flip(user_frame_original, 1) # 鏡像網路攝影機影像
        
        processed_user_frame, processed_ref_frame, similarity, is_done = \
            self.pose_comparator.process_user_frame(user_frame_flipped, self.target_display_height)

        self.display_image(processed_user_frame, self.video_label_user)
        self.display_image(processed_ref_frame, self.video_label_ref)

        self.lbl_similarity_score.setText(f"即時相似度: {similarity:.2f}")
        current_frame_num = self.pose_comparator.get_current_frame_num()
        total_ref_frames = self.pose_comparator.get_reference_frame_count()
        self.lbl_frame_count.setText(f"影格: {current_frame_num}/{total_ref_frames}")

        if is_done:
            self.stop_comparison(finished_naturally=True)


    def display_image(self, img_cv, label_widget: QLabel):
        if img_cv is None or img_cv.size == 0:
            # Create a black placeholder if image is invalid
            ph_h = label_widget.height() if label_widget.height() > 20 else self.target_display_height
            ph_w = label_widget.width() if label_widget.width() > 20 else int(ph_h * (16/9.0)) # Assume 16:9 if width unknown
            placeholder = np.zeros((ph_h, ph_w, 3), dtype=np.uint8)
            h, w, ch = placeholder.shape
            q_img = QImage(placeholder.data, w, h, ch * w, QImage.Format_RGB888)
        else:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        label_size = label_widget.size()
        # Ensure label_size is valid before scaling
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)
        else: # Fallback if label size is not yet determined (e.g., initial display)
            label_widget.setPixmap(pixmap.scaledToHeight(self.target_display_height, Qt.SmoothTransformation))


    def closeEvent(self, event):
        self.stop_comparison()
        if self.pose_comparator:
            self.pose_comparator.close() 
        event.accept()

if __name__ == '__main__':
    try:
        import mediapipe as mp
        mp_pose_test = mp.solutions.pose.Pose() 
        mp_pose_test.close()
    except ImportError:
        print("錯誤: mediapipe 函式庫未安裝。請安裝: pip install mediapipe")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤初始化 mediapipe: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    # Try to set a modern style
    available_styles = QStyleFactory.keys()
    if "Fusion" in available_styles:
        app.setStyle(QStyleFactory.create("Fusion"))
    elif "WindowsVista" in available_styles: # For Windows
        app.setStyle(QStyleFactory.create("WindowsVista"))

    main_win = DanceAppGUI()
    main_win.show()
    sys.exit(app.exec_())
