import cv2
import mediapipe as mp
import numpy as np
import os

# 輔助相似度計算函數
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
    def __init__(self, ref_video_path, similarity_method='cosine'):
        self.ref_video_path = ref_video_path
        self.similarity_method = similarity_method.lower()

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,        # 處理影片影格
            model_complexity=1,             # 模型複雜度
            smooth_landmarks=True,          # 平滑關鍵點
            min_detection_confidence=0.5,   # 最低偵測可信度
            min_tracking_confidence=0.5     # 最低追蹤可信度
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.reference_results_timeline = []  # 儲存參考影片的 MediaPipe results 物件
        self.reference_frames, self.reference_keypoints_timeline = self._process_reference_video()
        self.current_ref_frame_index = 0

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

    def _process_reference_video(self):
        """處理參考影片，提取每幀的影像和關鍵點。"""
        cap_ref = cv2.VideoCapture(self.ref_video_path)
        frames_timeline = []
        keypoints_timeline = []
        
        print("Processing reference video...")
        frame_idx = 0
        while cap_ref.isOpened():
            ret, frame = cap_ref.read()
            if not ret:
                break
            
            frames_timeline.append(frame.copy())
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # 優化效能
            results = self.pose_detector.process(image_rgb)
            image_rgb.flags.writeable = True
            
            self.reference_results_timeline.append(results) # 儲存完整的 results 物件
            keypoints = self._extract_keypoints_from_results(results)
            keypoints_timeline.append(keypoints)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} reference frames...")

        cap_ref.release()
        if not frames_timeline:
            print("Warning: Reference video is empty or could not be read.")
        else:
            print(f"Finished processing reference video: {len(frames_timeline)} frames total.")
        return frames_timeline, keypoints_timeline

    def _resize_frame(self, frame, target_height):
        """將影格大小調整至目標高度，同時保持寬高比。"""
        h, w = frame.shape[:2]
        if h == 0: return frame # Avoid division by zero if frame is invalid
        if h == target_height:
            return frame
        scale = target_height / h
        return cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)

    def calculate_similarity_score(self, kps_user_flat, kps_ref_flat):
        """根據選擇的方法計算相似度分數。"""
        if np.all(kps_user_flat == 0) or np.all(kps_ref_flat == 0):
            return 0.0  # 若任一姿勢未偵測到，相似度為0

        if self.similarity_method == 'cosine':
            sim = cosine_similarity_calc(kps_user_flat, kps_ref_flat)
            # 餘弦相似度範圍為 [-1, 1]。對於姿態向量（通常為非負），範圍通常為 [0, 1]。
            # 將其標準化到 [0, 1] 範圍作為分數。
            return max(0, sim) 
        elif self.similarity_method == 'euclidean':
            dist = euclidean_distance(kps_user_flat, kps_ref_flat)
            # 將歐氏距離轉換為相似度分數 (0 到 1，越高越好)
            # 這種標準化方法比較簡單，可能需要根據數據調整
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unsupported similarity method: {self.similarity_method}")

    def run_comparison(self):
        """執行即時姿態比較。"""
        cap_user = cv2.VideoCapture(0)  # 開啟即時攝影機
        if not cap_user.isOpened():
            print("Error: Could not open camera.")
            return

        if not self.reference_frames:
            print("Error: Reference video not processed or empty.")
            cap_user.release()
            self.pose_detector.close()
            return

        total_similarity_score = 0.0
        num_comparisons = 0
        
        # 獲取攝影機畫面尺寸作為目標顯示尺寸
        ret_user, frame_user_example = cap_user.read()
        if not ret_user:
            print("Error: Could not read initial frame from camera.")
            cap_user.release()
            self.pose_detector.close()
            return
        target_h, _ = frame_user_example.shape[:2]
        # cap_user.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置攝影機擷取 (對網路攝影機可能無效)

        print("Starting real-time comparison. Press ESC to quit.")
        while cap_user.isOpened() and self.current_ref_frame_index < len(self.reference_frames):
            ret_user, frame_user = cap_user.read()
            if not ret_user:
                print("Warning: Lost camera signal or stream ended.")
                break

            # 獲取當前參考影格及其關鍵點
            frame_ref_original = self.reference_frames[self.current_ref_frame_index]
            kps_ref_matrix = self.reference_keypoints_timeline[self.current_ref_frame_index]

            # 調整參考影格大小以進行並排顯示
            frame_ref_display = self._resize_frame(frame_ref_original.copy(), target_height=target_h)

            # 處理使用者影格以獲取關鍵點
            image_user_rgb = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
            image_user_rgb.flags.writeable = False
            results_user = self.pose_detector.process(image_user_rgb)
            image_user_rgb.flags.writeable = True
            kps_user_matrix = self._extract_keypoints_from_results(results_user)

            # 準備用於相似度計算的關鍵點 (展平的 x,y 座標)
            kps_user_flat = self._get_relevant_keypoints_for_similarity(kps_user_matrix)
            kps_ref_flat = self._get_relevant_keypoints_for_similarity(kps_ref_matrix)

            # 計算相似度
            current_similarity = 0.0
            # 僅在兩者皆偵測到姿態時計算並累加分數
            if not (np.all(kps_user_flat == 0) or np.all(kps_ref_flat == 0)):
                 current_similarity = self.calculate_similarity_score(kps_user_flat, kps_ref_flat)
                 total_similarity_score += current_similarity
                 num_comparisons += 1
            
            # --- 視覺化 ---
            # 在使用者影格上繪製骨架
            if results_user.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_user, results_user.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # 在參考影格上繪製骨架 (使用預存的 results 物件)
            ref_results_for_drawing = self.reference_results_timeline[self.current_ref_frame_index]
            if ref_results_for_drawing and ref_results_for_drawing.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_ref_display, ref_results_for_drawing.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # 參考骨架用綠色
                    self.mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2)
                )

            # 在使用者影格上顯示即時相似度分數
            cv2.putText(frame_user, f"Similarity: {current_similarity:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_user, f"Frame: {self.current_ref_frame_index + 1}/{len(self.reference_frames)}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 合併影格以並排顯示 (確保高度一致)
            if frame_user.shape[0] != frame_ref_display.shape[0]:
                 # 如果高度不匹配 (理論上不應發生，因 frame_ref_display 已調整大小)
                 # 作為備案，調整 user_frame (雖然不太理想)
                 frame_user_resized = self._resize_frame(frame_user, target_height=frame_ref_display.shape[0])
                 combined_display = np.hstack((frame_user_resized, frame_ref_display))
            else:
                 combined_display = np.hstack((frame_user, frame_ref_display))
            
            cv2.imshow('Pose Comparison (User | Reference)', combined_display)

            self.current_ref_frame_index += 1

            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 鍵結束 (waitKey(1) 使其盡快處理下一幀)
                print("Exiting...")
                break
        
        # 清理資源
        cap_user.release()
        cv2.destroyAllWindows()
        self.pose_detector.close()

        # 產生最終報告
        if num_comparisons > 0:
            overall_similarity = total_similarity_score / num_comparisons
            print(f"\n--- Movement Evaluation ---")
            print(f"Overall Similarity Score: {overall_similarity:.2f} (using {self.similarity_method} method)")
            self._generate_evaluation_report(overall_similarity)
        else:
            print("No valid comparisons made, cannot calculate overall similarity.")
            if not self.reference_frames:
                 print("Reason: Reference video not loaded correctly.")
            elif len(self.reference_frames) > 0 and self.current_ref_frame_index == 0:
                 print("Reason: No frames processed from live camera, or comparison loop did not run.")
            else:
                 print(f"Reason: Processed {self.current_ref_frame_index} frames, but no valid pose pairs for comparison.")

    def _generate_evaluation_report(self, overall_similarity):
        """根據整體相似度分數產生評估報告文字。"""
        print("\n--- Evaluation Report ---")
        print(f"Final Score: {overall_similarity:.2f}")
        if overall_similarity > 0.85:
            print("Evaluation: Excellent! Your movements are very similar to the reference video.")
        elif overall_similarity > 0.70:
            print("Evaluation: Good! Your movements are quite similar, but there's room for improvement.")
        elif overall_similarity > 0.50:
            print("Evaluation: Fair. Noticeable differences exist. Focus on matching poses more accurately.")
        else:
            print("Evaluation: Needs Improvement. Significant differences detected. Please carefully review the reference movements.")
        # TODO: 可根據每個關鍵點的分析或特定問題影格添加改進建議。

if __name__ == '__main__':
    # 提示使用者輸入參考影片的路徑
    ref_video_file = input("Please enter the path to the reference video (e.g., D:/videos/dance_move.mp4): ")

    if not os.path.exists(ref_video_file):
        print(f"Error: Reference video '{ref_video_file}' not found. Please ensure the path is correct.")
    else:
        # 選擇相似度計算方法 ('cosine' 或 'euclidean')
        # method = 'cosine' 
        method = 'euclidean' # 可切換至歐氏距離

        print(f"Using '{method}' method for similarity calculation.")
        comparator = PoseComparator(ref_video_path=ref_video_file, similarity_method=method)
        comparator.run_comparison()
