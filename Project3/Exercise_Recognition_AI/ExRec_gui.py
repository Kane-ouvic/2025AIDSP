import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QLineEdit, QSpinBox, QHBoxLayout, QMessageBox,QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, pyqtSlot # <<< ADD
from PyQt5.QtGui import QFont, QImage, QPixmap
from threading import Thread
import time # For time.sleep

import mediapipe as mp
import tensorflow as tf
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

# Suppress TF/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau # Removed ModelCheckpoint for brevity
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     Input, Flatten, Bidirectional, Permute, multiply)
class VideoUpdateBridge(QObject):
    frame_ready = pyqtSignal(QPixmap)

class VideoDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(640, 480)
        self.setStyleSheet("background-color: black; color:white;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Video Feed Here")


    def update_pixmap(self, pixmap):
        # Scale pixmap to fit label while maintaining aspect ratio
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ExerciseDecoder:
    def __init__(self, actions=None, data_folder_name='data',
                 num_sequences_per_action=50, sequence_length_frames=30,
                 collection_start_folder=1, test_split_ratio=0.1,
                 val_split_ratio_from_train=1/6, display_label_widget=None): # Renamed for clarity

        self.actions = np.array(actions) if actions is not None else np.array(['curl', 'press', 'squat'])
        self.num_classes = len(self.actions)

        self.DATA_PATH = os.path.join(os.getcwd(), data_folder_name)
        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)

        self.num_sequences_per_action = num_sequences_per_action
        self.sequence_length_frames = sequence_length_frames
        self.collection_start_folder = collection_start_folder

        self.num_landmarks = 33
        self.num_values_per_landmark = 4
        self.num_input_values = self.num_landmarks * self.num_values_per_landmark

        self._initialize_mediapipe()

        self.label_map = {label: num for num, label in enumerate(self.actions)}
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        if self.num_classes > len(self.colors):
            for i in range(self.num_classes - len(self.colors)):
                self.colors.append(tuple(np.random.randint(0, 255, size=3).tolist()))

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio_from_train = val_split_ratio_from_train

        self.models = {}
        self.eval_results = {}

        self.counter = {act:0 for act in self.actions}
        # Removed redundant individual counters, use self.counter[action_name]
        self.curl_stage, self.press_stage, self.squat_stage = None, None, None # Keep stages for logic

        self.display_label_widget = display_label_widget # Store the VideoDisplay widget
        self.bridge = VideoUpdateBridge()
        if self.display_label_widget:
            self.bridge.frame_ready.connect(self.display_label_widget.update_pixmap)

        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _qt_display_image(self, image):
        if self._stop_requested:
            return
        # Ensure image is BGR
        if image.ndim == 2: image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else: image_bgr = image

        height, width, channel = image_bgr.shape
        bytes_per_line = channel * width
        q_img = QImage(image_bgr.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.bridge.frame_ready.emit(pixmap.copy())

    def _initialize_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def _mediapipe_detection(self, image, pose_model):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_model.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr, results

    def _draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def _extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(self.num_input_values)
        return pose

    def setup_data_folders(self):
        for action in self.actions:
            for sequence_idx in range(self.collection_start_folder,
                                      self.collection_start_folder + self.num_sequences_per_action):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence_idx)))
                except FileExistsError:
                    pass
        print(f"Data folders set up in: {self.DATA_PATH}")

    def collect_training_data(self, cap_source=0): # Removed break_key
        self._stop_requested = False
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"[collect_training_data]: Error: Could not open video source {cap_source}")
            return
        try:
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
                for action_idx, action_name in enumerate(self.actions):
                    if self._stop_requested: break
                    print(f"Preparing for {action_name}")
                    for i in range(3,0,-1): # Countdown
                        if self._stop_requested: break
                        ret_c, frame_c = cap.read()
                        if not ret_c: break
                        img_c = frame_c.copy()
                        cv2.putText(img_c, f"Starting in {i}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        self._qt_display_image(img_c)
                        time.sleep(0.5) # Give GUI time to update
                    if self._stop_requested or not cap.isOpened() or (locals().get('ret_c') and not ret_c): break
                    for sequence_num in range(self.collection_start_folder,
                                              self.collection_start_folder + self.num_sequences_per_action):
                        


                        for frame_num in range(self.sequence_length_frames):
                            if self._stop_requested: break
                            ret, frame = cap.read()
                            if not ret:
                                print("Error: Failed to read frame from camera.")
                                break

                            image, results = self._mediapipe_detection(frame, pose_model)
                            self._draw_landmarks(image, results)
                            display_text = f'Rec: {action_name} Vid#{sequence_num} Frm#{frame_num+1}'
                            color_idx = action_idx % len(self.colors)
                            cv2.putText(image, display_text, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                            cv2.putText(image, display_text, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[color_idx], 1, cv2.LINE_AA)
                            if frame_num == 0:
                                 cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

                            self._qt_display_image(image)
                            keypoints = self._extract_keypoints(results)
                            npy_path = os.path.join(self.DATA_PATH, action_name, str(sequence_num), str(frame_num))
                            np.save(npy_path, keypoints)
                            QApplication.processEvents() # Allow Qt to process events
                        if self._stop_requested: break
                        print(f"Finished collecting {action_name}, Video #{sequence_num}")
                    if self._stop_requested: break
        finally:
            if cap.isOpened(): cap.release()
        print("collect_training_data finished or stopped.")


    def run_real_time_inference(self, model_name_to_use, cap_source=0, threshold=0.5,
                                save_video=False, output_video_filename=None): # Removed break_key
        self._stop_requested = False
        if model_name_to_use not in self.models or self.models[model_name_to_use] is None:
            if not self.load_model(model_name_to_use):
                print(f"Failed to load model {model_name_to_use}")
                return
        active_model = self.models[model_name_to_use]
        self.counter = {act:0 for act in self.actions} # Reset counters
        self.curl_stage, self.press_stage, self.squat_stage = None, None, None
        sequence_data = []
        current_action_prediction = ''
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {cap_source}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = None
        if save_video:
            if output_video_filename is None: output_video_filename = f"{model_name_to_use}_real_time_test.avi"
            video_path = os.path.join(os.getcwd(), output_video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps_vid = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            out_video = cv2.VideoWriter(video_path, fourcc, fps_vid, (frame_width, frame_height))
            print(f"Saving video to {video_path}")
        try:
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
                while cap.isOpened() and not self._stop_requested:
                    ret, frame = cap.read()
                    if not ret: break
                    image, results = self._mediapipe_detection(frame, pose_model)
                    self._draw_landmarks(image, results)
                    keypoints = self._extract_keypoints(results)
                    sequence_data.append(keypoints)
                    sequence_data = sequence_data[-self.sequence_length_frames:]

                    if len(sequence_data) == self.sequence_length_frames:
                        prediction_probs = active_model.predict(np.expand_dims(sequence_data, axis=0), verbose=0)[0]
                        predicted_action_idx = np.argmax(prediction_probs)
                        confidence = prediction_probs[predicted_action_idx]
                        action_color = (50,50,50)
                        current_action_prediction = '' # Reset
                        if confidence >= threshold and predicted_action_idx < len(self.actions):
                            current_action_prediction = self.actions[predicted_action_idx]
                            color_idx = predicted_action_idx % len(self.colors)
                            action_color = self.colors[color_idx]
                        cv2.rectangle(image, (0,0), (frame_width, 40), action_color, -1)
                        image = self._prob_viz(prediction_probs, image)
                        if results.pose_landmarks and current_action_prediction:
                            try:
                                self._count_reps_logic(image, current_action_prediction,
                                                      results.pose_landmarks.landmark, frame_width, frame_height)
                            except Exception as e: print(f"Rep count error: {e}")
                    # Display rep counts - simplified
                    for i, act_name in enumerate(self.actions[:3]): # Display max 3
                        cv2.putText(image, f'{act_name[:1]}: {self.counter.get(act_name,0)}', (15 + i*100,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

                    self._qt_display_image(image)
                    if save_video and out_video is not None: out_video.write(image)
                    QApplication.processEvents() # Allow Qt to process events
        finally:
            if cap.isOpened(): cap.release()
            if out_video is not None: out_video.release()
        print("run_real_time_inference finished or stopped.")

    # --- Placeholder for other ExerciseDecoder methods (preprocess, train, models, etc.) ---
    # --- These methods were present in your original code and should be included here ---
    # --- For brevity, I'm omitting them, but they are NECESSARY for full functionality ---
    def preprocess_data(self):
        # ... (Your existing preprocess_data logic) ...
        print("Preprocessing data...")
        sequences, labels = [], []
        for action_name in self.actions:
            action_path = os.path.join(self.DATA_PATH, action_name)
            if not os.path.exists(action_path):
                print(f"Warning: Action path {action_path} does not exist. Skipping.")
                continue

            sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
            valid_sequence_ids = []
            for f_id in sequence_folders:
                if f_id.isdigit(): # Process only if folder name is a number
                    valid_sequence_ids.append(int(f_id))
            
            for sequence_id in sorted(valid_sequence_ids): 
                window = []
                sequence_path = os.path.join(action_path, str(sequence_id))
                if not os.path.exists(sequence_path): continue # Skip if path somehow invalid
                
                num_frames_in_sequence = len([name for name in os.listdir(sequence_path) if name.endswith('.npy')])
                if num_frames_in_sequence != self.sequence_length_frames:
                    # print(f"Warning: Seq {action_name}/{sequence_id} has {num_frames_in_sequence} frames, expected {self.sequence_length_frames}. Skipping.")
                    continue

                all_frames_loaded = True
                for frame_num in range(self.sequence_length_frames):
                    res_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    if not os.path.exists(res_path):
                        # print(f"Warning: Frame {frame_num}.npy not found in {sequence_path}. Skipping sequence {sequence_id}.")
                        all_frames_loaded = False
                        break
                    try:
                        res = np.load(res_path)
                        window.append(res)
                    except Exception as e: # Catch potential loading errors
                        print(f"Error loading {res_path}: {e}")
                        all_frames_loaded = False
                        break
                
                if all_frames_loaded and len(window) == self.sequence_length_frames: 
                    sequences.append(window)
                    labels.append(self.label_map[action_name])

        if not sequences:
            print("Error: No sequences loaded. Cannot proceed with preprocessing.")
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = [None]*6
            return

        X = np.array(sequences)
        y = to_categorical(labels, num_classes=self.num_classes).astype(int)
        print(f"Original data shape: X={X.shape}, y={y.shape}")

        # Simplified splitting if dataset is too small for stratification
        min_samples_for_stratify = 2 * self.num_classes 
        can_stratify = X.shape[0] >= min_samples_for_stratify and all(np.sum(y, axis=0) > 1)


        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_split_ratio, random_state=1, 
                             stratify=y if can_stratify else None)
        
        if self.X_train.shape[0] > 1 : # Need at least 2 samples to split for validation
            can_stratify_train = self.X_train.shape[0] >= min_samples_for_stratify and (self.y_train.ndim > 1 and all(np.sum(self.y_train, axis=0) > 1))

            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(self.X_train, self.y_train, test_size=self.val_split_ratio_from_train, 
                                 random_state=2, stratify=self.y_train if can_stratify_train else None)
        else:
            print("Warning: Not enough training data for validation split. Using training data as validation.")
            self.X_val, self.y_val = self.X_train.copy() if self.X_train.size > 0 else None, \
                                     self.y_train.copy() if self.y_train.size > 0 else None


        print(f"Training data shape: X_train={self.X_train.shape if self.X_train is not None else 'None'}, y_train={self.y_train.shape if self.y_train is not None else 'None'}")
        print(f"Validation data shape: X_val={self.X_val.shape if self.X_val is not None else 'None'}, y_val={self.y_val.shape if self.y_val is not None else 'None'}")
        print(f"Test data shape: X_test={self.X_test.shape if self.X_test is not None else 'None'}, y_test={self.y_test.shape if self.y_test is not None else 'None'}")


    def _get_training_callbacks(self, model_name):
        log_dir = os.path.join(os.getcwd(), 'logs', f"{model_name}-{int(time.time())}", '')
        tb_callback = TensorBoard(log_dir=log_dir)
        es_callback = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10, verbose=1, mode='min')
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1, mode='min')
        return [tb_callback, es_callback, lr_callback]

    def _build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(self.sequence_length_frames, self.num_input_values)))
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=False, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def _attention_block(self, inputs, time_steps):
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def _build_attn_lstm_model(self, hidden_units=128): # Changed to 128 to match typical use
        inputs_layer = Input(shape=(self.sequence_length_frames, self.num_input_values))
        lstm_out = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs_layer)
        attention_mul = self._attention_block(lstm_out, self.sequence_length_frames)
        attention_mul = Flatten()(attention_mul)
        x = Dense(2 * hidden_units, activation='relu')(attention_mul)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=[inputs_layer], outputs=x)
        return model

    def build_and_compile_models(self, model_name, learning_rate=0.001):
        try:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except AttributeError: # Fallback for older TF
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)


        if model_name == 'LSTM':
            print(f"[build_and_compile_models]: Building {model_name} model...")
            self.models[model_name] = self._build_lstm_model()
        elif model_name == 'LSTM_Attention_128HUs':
            print(f"[build_and_compile_models]: Building LSTM_Attention_128HUs model...") # Corrected f-string
            self.models['LSTM_Attention_128HUs'] = self._build_attn_lstm_model(hidden_units=128)
        else:
            print(f"Error: Unknown model name '{model_name}'")
            return

        if self.num_classes == 0:
            print("Error: num_classes is 0. Cannot compile model.")
            return

        self.models[model_name].compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        print(f"[build_and_compile_models]: {model_name} model built and compiled.")
        self.models[model_name].summary(print_fn=lambda x: print(x) if "Total params" in x or "Trainable params" in x or "Layer (type)" in x else None)

    def train_model(self, model_name, epochs=50, batch_size=32):
        if model_name not in self.models:
            print(f"[train_model]: Error: Model {model_name} not found.")
            return
        if self.X_train is None or self.y_train is None or self.X_train.size == 0:
            print("[train_model]: Error: Training data not available or empty.")
            return
        
        validation_data_to_use = None
        if self.X_val is not None and self.y_val is not None and self.X_val.size > 0:
            validation_data_to_use = (self.X_val, self.y_val)
        else:
            print("[train_model]: Warning: Validation data not available or empty. Training without validation.")


        model = self.models[model_name]
        callbacks = self._get_training_callbacks(model_name)
        print(f"[train_model]: Training {model_name}...")
        history = model.fit(self.X_train, self.y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=validation_data_to_use,
                            callbacks=callbacks,
                            verbose=1)
        print(f"{model_name} training finished.")
        return history

    def save_model(self, model_name, filename=None):
        if model_name not in self.models:
            print(f"[save_model]: Model {model_name} not found.")
            return
        if filename is None: filename = f"{model_name}.h5"
        save_path = os.path.join(self.DATA_PATH, filename)
        self.models[model_name].save(save_path)
        print(f"[save_model]: Model {model_name} saved to {save_path}")

    def load_model(self, model_name, filename=None, compile_model=True):
        if filename is None: filename = f"{model_name}.h5"
        load_path = os.path.join(self.DATA_PATH, filename)
        if not os.path.exists(load_path):
            print(f"[load_model]: Error: File {load_path} not found.")
            return False
        try:
            # Custom objects might be needed if you have custom layers/activations not standard in Keras
            custom_objects = {} # Add custom layers here if needed, e.g., {'AttentionLayer': AttentionLayer}
            loaded_model = tf.keras.models.load_model(load_path, custom_objects=custom_objects, compile=compile_model)
            self.models[model_name] = loaded_model
            print(f"[load_model]: Model {model_name} loaded from {load_path}")
            return True
        except Exception as e:
            print(f"[load_model]: Error loading model {model_name} from {load_path}: {e}")
            return False

    def evaluate_all_models(self):
        if self.X_test is None or self.y_test is None or self.X_test.size == 0:
            print("Error: Test data not available or empty.")
            return
        self.eval_results = {'confusion_matrix': {}, 'accuracy': {}, 'precision': {}, 'recall': {}, 'f1_score': {}}
        y_true_labels = np.argmax(self.y_test, axis=1).tolist()
        for model_name, model_instance in self.models.items():
            if model_instance is None: continue
            print(f"\nEvaluating {model_name}:")
            try:
                y_pred_probs = model_instance.predict(self.X_test, verbose=0)
            except Exception as e: print(f"Pred err {model_name}: {e}"); continue
            y_pred_labels = np.argmax(y_pred_probs, axis=1).tolist()
            cm = multilabel_confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(self.num_classes)))
            self.eval_results['confusion_matrix'][model_name] = cm; print(f"CM:\n{cm}")
            acc = accuracy_score(y_true_labels, y_pred_labels)
            self.eval_results['accuracy'][model_name] = acc; print(f"Acc: {acc:.4f}")
            report = classification_report(y_true_labels, y_pred_labels, target_names=self.actions,
                                           labels=list(range(self.num_classes)), output_dict=True, zero_division=0)
            self.eval_results['precision'][model_name] = report['weighted avg']['precision']
            self.eval_results['recall'][model_name] = report['weighted avg']['recall']
            self.eval_results['f1_score'][model_name] = report['weighted avg']['f1-score']
            print(f"P: {report['weighted avg']['precision']:.4f}, R: {report['weighted avg']['recall']:.4f}, F1: {report['weighted avg']['f1-score']:.4f}")
            print(f"Report:\n{classification_report(y_true_labels, y_pred_labels, target_names=self.actions, labels=list(range(self.num_classes)), zero_division=0)}")
        return self.eval_results

    def _calculate_angle(self, a, b, c):
        a,b,c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360 - angle if angle > 180.0 else angle

    def _get_coordinates(self, landmark_list, side, joint_name):
        try:
            coord_idx = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_{joint_name.upper()}").value
            lm = landmark_list[coord_idx]
            return [lm.x, lm.y]
        except (AttributeError, IndexError) as e:
            # print(f"Error getting coord for {side}_{joint_name}: {e}")
            raise ValueError(f"Invalid landmark access: {side.upper()}_{joint_name.upper()}")


    def _viz_joint_angle(self, image, angle, joint_coords_normalized, frame_width, frame_height):
        joint_pixel_coords = tuple(np.multiply(joint_coords_normalized, [frame_width, frame_height]).astype(int))
        cv2.putText(image, str(int(angle)), joint_pixel_coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def _count_reps_logic(self, image, current_action, landmark_list, frame_width, frame_height):
        if current_action == 'curl':
            try:
                shoulder = self._get_coordinates(landmark_list, 'left', 'shoulder')
                elbow = self._get_coordinates(landmark_list, 'left', 'elbow')
                wrist = self._get_coordinates(landmark_list, 'left', 'wrist')
                angle = self._calculate_angle(shoulder, elbow, wrist)
                if angle < 30: self.curl_stage = "up"
                if angle > 140 and self.curl_stage == 'up':
                    self.curl_stage = "down"; self.counter['curl'] += 1 # Use generic counter
                self.press_stage, self.squat_stage = None, None
                self._viz_joint_angle(image, angle, elbow, frame_width, frame_height)
            except ValueError as e: print(f"Curl coord error: {e}")
        elif current_action == 'press':
            try:
                shoulder = self._get_coordinates(landmark_list, 'left', 'shoulder')
                elbow = self._get_coordinates(landmark_list, 'left', 'elbow')
                wrist = self._get_coordinates(landmark_list, 'left', 'wrist')
                elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                shoulder2elbow_dist = math.dist(shoulder, elbow)
                shoulder2wrist_dist = math.dist(shoulder, wrist)
                if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist): self.press_stage = "up"
                if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage == 'up'):
                    self.press_stage = 'down'; self.counter['press'] += 1 # Use generic counter
                self.curl_stage, self.squat_stage = None, None
                self._viz_joint_angle(image, elbow_angle, elbow, frame_width, frame_height)
            except ValueError as e: print(f"Press coord error: {e}")
        elif current_action == 'squat':
            try:
                l_s = self._get_coordinates(landmark_list, 'left', 'shoulder'); l_h = self._get_coordinates(landmark_list, 'left', 'hip')
                l_k = self._get_coordinates(landmark_list, 'left', 'knee'); l_a = self._get_coordinates(landmark_list, 'left', 'ankle')
                r_s = self._get_coordinates(landmark_list, 'right', 'shoulder'); r_h = self._get_coordinates(landmark_list, 'right', 'hip')
                r_k = self._get_coordinates(landmark_list, 'right', 'knee'); r_a = self._get_coordinates(landmark_list, 'right', 'ankle')
                lk_ang = self._calculate_angle(l_h,l_k,l_a); rk_ang = self._calculate_angle(r_h,r_k,r_a)
                lh_ang = self._calculate_angle(l_s,l_h,l_k); rh_ang = self._calculate_angle(r_s,r_h,r_k)
                thr = 165
                if (lk_ang<thr and rk_ang<thr and lh_ang<thr and rh_ang<thr): self.squat_stage="down"
                if (lk_ang>thr and rk_ang>thr and lh_ang>thr and rh_ang>thr and self.squat_stage=='down'):
                    self.squat_stage='up'; self.counter['squat'] +=1 # Use generic counter
                self.curl_stage, self.press_stage = None,None
                self._viz_joint_angle(image, lk_ang,l_k,frame_width,frame_height); self._viz_joint_angle(image, lh_ang,l_h,frame_width,frame_height)
            except ValueError as e: print(f"Squat coord error: {e}")
        else:
            self.curl_stage, self.press_stage, self.squat_stage = None,None,None


    def _prob_viz(self, res_probs, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res_probs):
            if num >= len(self.actions): continue
            action_name = self.actions[num]
            color_idx = num % len(self.colors)
            color = self.colors[color_idx]
            bar_start_x, bar_start_y = 0, 60 + num * 40
            bar_end_x, bar_end_y = int(prob * 100), 90 + num * 40 # Scale prob to 100px bar
            cv2.rectangle(output_frame, (bar_start_x, bar_start_y), (bar_end_x, bar_end_y), color, -1)
            cv2.putText(output_frame, action_name, (10, 85 + num * 40), # x=10 for padding
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame
# ... Rest of ExerciseDecoder (model definitions, training, saving, loading, eval) ...

class FitnessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.video_display_widget = VideoDisplay()
        self.decoder = None
        self.current_thread = None
        self.actions_text_list_config = ['curl', 'press', 'squat']
        self.num_sequences_val_config = 20
        self.init_ui()
        self.apply_config_and_init_decoder() # Initial configuration

    def init_ui(self):
        self.setWindowTitle("Smart Fitness Tracker")
        self.setGeometry(100, 100, 950, 600)
        overall_layout = QHBoxLayout(self) # Set layout directly on self

        left_pane_widget = QWidget()
        left_pane_layout = QVBoxLayout(left_pane_widget)
        left_pane_layout.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("🏋️‍♀️ Smart Fitness App")
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        left_pane_layout.addWidget(self.title_label)
        left_pane_layout.addSpacing(15)

        self.actions_label = QLabel("Actions (comma-separated):")
        self.actions_input = QLineEdit()
        self.actions_input.setText(",".join(self.actions_text_list_config))
        self.actions_input.textChanged.connect(self.ui_actions_changed) # Connect
        actions_row_layout = QHBoxLayout()
        actions_row_layout.addWidget(self.actions_label); actions_row_layout.addWidget(self.actions_input)
        left_pane_layout.addLayout(actions_row_layout)

        self.num_seq_label = QLabel("Sequences per action:")
        self.num_seq_spinbox = QSpinBox()
        self.num_seq_spinbox.setRange(1, 100); self.num_seq_spinbox.setValue(self.num_sequences_val_config)
        self.num_seq_spinbox.valueChanged.connect(self.ui_num_sequences_changed) # Connect
        sequences_row_layout = QHBoxLayout()
        sequences_row_layout.addWidget(self.num_seq_label); sequences_row_layout.addWidget(self.num_seq_spinbox)
        sequences_row_layout.addStretch(1)
        left_pane_layout.addLayout(sequences_row_layout)
        left_pane_layout.addSpacing(15)

        fixed_button_width = 180
        self.config_btn = QPushButton("Apply Config / Init Decoder")
        self.config_btn.setFixedWidth(fixed_button_width)
        self.config_btn.clicked.connect(self.apply_config_and_init_decoder)
        left_pane_layout.addWidget(self.config_btn, alignment=Qt.AlignLeft)

        self.finetune_btn = QPushButton("Fine Tune (Collect & Train)")
        self.finetune_btn.setFixedWidth(fixed_button_width)
        self.finetune_btn.clicked.connect(self.start_finetune_thread)
        left_pane_layout.addWidget(self.finetune_btn, alignment=Qt.AlignLeft)

        self.start_workout_btn = QPushButton("Start Workout")
        self.start_workout_btn.setFixedWidth(fixed_button_width)
        self.start_workout_btn.clicked.connect(self.start_workout_thread)
        left_pane_layout.addWidget(self.start_workout_btn, alignment=Qt.AlignLeft)

        self.stop_btn = QPushButton("Stop Current Action")
        self.stop_btn.setFixedWidth(fixed_button_width)
        self.stop_btn.clicked.connect(self.stop_decoder_action)
        self.stop_btn.setEnabled(False)
        left_pane_layout.addWidget(self.stop_btn, alignment=Qt.AlignLeft)

        left_pane_layout.addStretch(1)
        overall_layout.addWidget(left_pane_widget, 1)
        overall_layout.addWidget(self.video_display_widget, 2) # Video display widget on the right
        self.setStyleSheet(self.load_styles())

    def ui_actions_changed(self, text):
        self.actions_text_list_config = [a.strip() for a in text.split(',') if a.strip()]
        print(f"UI Actions to be applied: {self.actions_text_list_config}")

    def ui_num_sequences_changed(self, value):
        self.num_sequences_val_config = value
        print(f"UI Num Sequences to be applied: {self.num_sequences_val_config}")

    def apply_config_and_init_decoder(self):
        if self.current_thread and self.current_thread.is_alive():
            if self.decoder: self.decoder.request_stop()
            self.current_thread.join(timeout=0.5)
            if self.current_thread.is_alive():
                QMessageBox.warning(self, "Busy", "Previous task running. Stop it first or wait.")
                return
            self._reset_thread_and_buttons()

        if not self.actions_text_list_config:
            QMessageBox.warning(self, "Config Error", "Actions list cannot be empty.")
            return

        self.decoder = ExerciseDecoder(
            actions=self.actions_text_list_config,
            num_sequences_per_action=self.num_sequences_val_config,
            display_label_widget=self.video_display_widget # Pass the widget
        )
        QMessageBox.information(self, "Configured", f"Decoder configured for: {', '.join(self.decoder.actions)}")
        self.video_display_widget.setText("Video Feed Here") # Reset

    def _start_thread_if_safe(self, target_func, args_tuple=()): # Default empty tuple for args
        if self.current_thread and self.current_thread.is_alive():
            QMessageBox.warning(self, "Busy", "Another action is running.")
            return False
        if not self.decoder:
            QMessageBox.warning(self, "Not Ready", "Decoder not initialized. Please Apply Config.")
            return False

        # Ensure decoder uses latest UI config if it was changed after last apply
        current_ui_actions = [a.strip() for a in self.actions_input.text().split(',') if a.strip()]
        current_ui_seq = self.num_seq_spinbox.value()
        if self.decoder.actions.tolist() != current_ui_actions or \
           self.decoder.num_sequences_per_action != current_ui_seq:
            print("Configuration changed, re-initializing decoder before starting thread.")
            self.apply_config_and_init_decoder() # Re-init with current UI values
            if not self.decoder: return False # If re-init failed

        self.target_for_thread_completion = target_func.__name__ # Store for later
        self.current_thread = Thread(target=self._thread_wrapper, args=(target_func, args_tuple), daemon=True)
        self.current_thread.start()
        self.config_btn.setEnabled(False)
        self.finetune_btn.setEnabled(False)
        self.start_workout_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        return True

    def _thread_wrapper(self, target_func, args_tuple):
        """Wraps the target function to call _on_thread_finished."""
        try:
            target_func(*args_tuple)
        except Exception as e:
            print(f"Error in thread for {target_func.__name__}: {e}")
        finally:
            # This is called from the worker thread. To update GUI safely,
            # we would ideally emit a signal. For simplicity with threading.Thread:
            QMetaObject.invokeMethod(self, "_on_thread_finished", Qt.QueuedConnection)

    @pyqtSlot() # <<< CRITICAL: Decorate the method
    def _on_thread_finished(self):
        """Slot to be called when a worker thread finishes."""
        print(f"Thread for '{getattr(self, 'target_for_thread_completion', 'unknown task')}' finished.")
        self._reset_thread_and_buttons()
        # If it was finetune data collection, now do the sync training part
        if hasattr(self, 'target_for_thread_completion') and self.target_for_thread_completion == 'collect_training_data':
            if self.decoder and not self.decoder._stop_requested: # Check if not stopped by user
                self.perform_synchronous_training()
            else:
                print("Data collection was stopped by user, skipping training.")
        # Clear the target attribute after use
        if hasattr(self, 'target_for_thread_completion'):
            delattr(self, 'target_for_thread_completion')


    def _reset_thread_and_buttons(self):
        self.current_thread = None
        self.config_btn.setEnabled(True)
        self.finetune_btn.setEnabled(True)
        self.start_workout_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


    def start_finetune_thread(self):
        if self._start_thread_if_safe(self.decoder.collect_training_data, (0,)):
            print("Data collection for fine-tuning started.")
            # Training part will be called by _on_thread_finished

    def perform_synchronous_training(self):
        print("Starting synchronous training phase...")
        QApplication.processEvents() # Update UI
        try:
            self.decoder.setup_data_folders()
            self.decoder.collection_start_folder += self.decoder.num_sequences_per_action
            self.decoder.preprocess_data()
            if self.decoder.X_train is None or self.decoder.X_train.size == 0:
               print("Preprocessing failed or no data. Training skipped.")
               return

            model_to_train = 'LSTM_Attention_128HUs'
            self.decoder.build_and_compile_models(model_name=model_to_train, learning_rate=0.001)
            print(f"Training {model_to_train}... GUI will be unresponsive.")
            QApplication.processEvents()
            self.decoder.train_model(model_to_train, epochs=10, batch_size=16) # This blocks
            self.decoder.save_model(model_to_train)
            print("Fine-tuning training phase complete.")
            QMessageBox.information(self, "Training Done", "Model training and saving finished.")
        except Exception as e:
            print(f"Error during synchronous training: {e}")
            QMessageBox.critical(self, "Training Error", f"Error: {e}")


    def start_workout_thread(self):
        weight_name = 'LSTM_Attention_128HUs'
        # No need to call apply_config_and_init_decoder here, _start_thread_if_safe will do it.
        if self.decoder and not self.decoder.load_model(weight_name): # load_model is usually fast
            QMessageBox.warning(self, "Model Load Failed", f"Could not load {weight_name}. Train first.")
            return
        if self._start_thread_if_safe(self.decoder.run_real_time_inference,
                                     (weight_name, 0, 0.7, False, None)):
            print("Workout (inference) thread started.")

    def stop_decoder_action(self):
        if self.decoder:
            self.decoder.request_stop()
            print("Stop request sent.")
            self.stop_btn.setEnabled(False) # Disable to prevent multi-clicks
            # Buttons re-enabled by _on_thread_finished

    def closeEvent(self, event):
        if self.decoder: self.decoder.request_stop()
        if self.current_thread and self.current_thread.is_alive():
            print("Joining thread on close...")
            self.current_thread.join(timeout=1.0)
        super().closeEvent(event)

    def load_styles(self): # Keep your styles
        return """
            QWidget { background-color: #f0f0f0; } QLabel { font-size: 14px; color: #333; }
            QLineEdit, QSpinBox {
                border: 1px solid #5F9EA0; border-radius: 3px; padding: 4px; font-size: 12px;
                background-color: #f8f8f8; color: #333;
            }
            QPushButton {
                background-color: #5cb85c; color: white; border: none;
                padding: 10px 15px; margin-top: 8px; border-radius: 4px; font-size: 14px;
            }
            QPushButton:hover { background-color: #4cae4c; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Need to import QMetaObject for invokeMethod
    from PyQt5.QtCore import QMetaObject
    window = FitnessApp()
    window.show()
    sys.exit(app.exec_())