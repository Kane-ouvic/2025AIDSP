import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QLineEdit, QSpinBox, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

## 0. Import Dependencies
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow as tf
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

# Suppress TF/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0) # Changed to 0 to reduce autograph warnings further
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     Input, Flatten, Bidirectional, Permute, multiply)

class ExerciseDecoder:
    def __init__(self, actions=None, data_folder_name='data', 
                 num_sequences_per_action=50, sequence_length_frames=30, 
                 collection_start_folder=1, test_split_ratio=0.1, 
                 val_split_ratio_from_train=1/6): # 1/6 matches notebook's 15/90 on 135 samples

        self.actions = np.array(actions) if actions is not None else np.array(['curl', 'press', 'squat'])
        self.num_classes = len(self.actions)
        
        self.DATA_PATH = os.path.join(os.getcwd(), data_folder_name)
        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)

        self.num_sequences_per_action = num_sequences_per_action
        self.sequence_length_frames = sequence_length_frames
        self.collection_start_folder = collection_start_folder

        self.num_landmarks = 33
        self.num_values_per_landmark = 4 # x, y, z, visibility
        self.num_input_values = self.num_landmarks * self.num_values_per_landmark

        self._initialize_mediapipe()

        self.label_map = {label: num for num, label in enumerate(self.actions)}
        # Colors in BGR for OpenCV
        self.colors = [(245,117,16), (117,245,16), (16,117,245)] 
        if self.num_classes > len(self.colors): # Add more colors if needed
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
        self.curl_counter, self.press_counter, self.squat_counter = 0, 0, 0
        self.curl_stage, self.press_stage, self.squat_stage = None, None, None

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

    def collect_training_data(self, cap_source=0, break_key='q'):
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"[collect_training_data]: Error: Could not open video source {cap_source}")
            return

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
            for action_idx, action_name in enumerate(self.actions):
                for sequence_num in range(self.collection_start_folder, 
                                          self.collection_start_folder + self.num_sequences_per_action):
                    print(f"Preparing for {action_name}, Video #{sequence_num}")
                    # for i in range(3, 0, -1): # Short countdown
                    #     ret, frame = cap.read()
                    #     if not ret: 
                    #         print("Error reading frame during countdown.")
                    #         cap.release()
                    #         cv2.destroyAllWindows()
                    #         return
                        
                    #     image_display = frame.copy()
                    #     cv2.putText(image_display, f'Prepare for {action_name}, Vid #{sequence_num}. Start in {i}',
                    #                 (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                    #     cv2.imshow('OpenCV Feed', image_display)
                    #     if cv2.waitKey(1000) & 0xFF == ord(break_key): # Wait 1 sec
                    #         print("Collection interrupted by user during prep.")
                    #         cap.release()
                    #         cv2.destroyAllWindows()
                    #         return
                    
                    for frame_num in range(self.sequence_length_frames):
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to read frame from camera.")
                            break 
                        
                        image, results = self._mediapipe_detection(frame, pose_model)
                        self._draw_landmarks(image, results)

                        display_text = f'Rec: {action_name} Vid#{sequence_num} Frm#{frame_num+1}'
                        color_idx = action_idx % len(self.colors) # Ensure color_idx is within bounds
                        
                        cv2.putText(image, display_text, (15,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA) # White background text
                        cv2.putText(image, display_text, (15,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[color_idx], 1, cv2.LINE_AA) # Color text
                        
                        if frame_num == 0:
                             cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                        
                        cv2.imshow('OpenCV Feed', image)

                        keypoints = self._extract_keypoints(results)
                        npy_path = os.path.join(self.DATA_PATH, action_name, str(sequence_num), str(frame_num))
                        np.save(npy_path, keypoints)

                        if cv2.waitKey(10) & 0xFF == ord(break_key):
                            print("Collection interrupted by user.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    print(f"Finished collecting {action_name}, Video #{sequence_num}")

        cap.release()
        cv2.destroyAllWindows()
        print("Data collection finished.")

    def preprocess_data(self):
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
            
            # Only consider sequences within the collected range if desired, or all found
            # For now, process all valid integer-named sequence folders
            for sequence_id in sorted(valid_sequence_ids): 
                window = []
                sequence_path = os.path.join(action_path, str(sequence_id))
                
                num_frames_in_sequence = len([name for name in os.listdir(sequence_path) if name.endswith('.npy')])
                if num_frames_in_sequence != self.sequence_length_frames:
                    print(f"Warning: Seq {action_name}/{sequence_id} has {num_frames_in_sequence} frames, expected {self.sequence_length_frames}. Skipping.")
                    continue

                all_frames_loaded = True
                for frame_num in range(self.sequence_length_frames):
                    res_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    if not os.path.exists(res_path):
                        print(f"Warning: Frame {frame_num}.npy not found in {sequence_path}. Skipping sequence {sequence_id}.")
                        all_frames_loaded = False
                        break
                    res = np.load(res_path)
                    window.append(res)
                
                if all_frames_loaded and window: 
                    sequences.append(window)
                    labels.append(self.label_map[action_name])

        if not sequences:
            print("Error: No sequences loaded. Cannot proceed with preprocessing.")
            return

        X = np.array(sequences)
        y = to_categorical(labels, num_classes=self.num_classes).astype(int)
        print(f"Original data shape: X={X.shape}, y={y.shape}")

        min_samples_for_stratify = np.min([np.sum(y[:, i]) for i in range(y.shape[1])])

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_split_ratio, random_state=1, 
                             stratify=y if min_samples_for_stratify > 1 else None)
        
        if len(self.X_train) > 1 :
            min_samples_train_for_stratify = np.min([np.sum(self.y_train[:, i]) for i in range(self.y_train.shape[1])])
            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(self.X_train, self.y_train, test_size=self.val_split_ratio_from_train, 
                                 random_state=2, stratify=self.y_train if min_samples_train_for_stratify > 1 else None)
        else:
            print("Warning: Not enough training data for validation split. Using training data as validation.")
            self.X_val, self.y_val = self.X_train, self.y_train # Fallback

        print(f"Training data shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"Validation data shape: X_val={self.X_val.shape}, y_val={self.y_val.shape}")
        print(f"Test data shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")

    def _get_training_callbacks(self, model_name):
        log_dir = os.path.join(os.getcwd(), 'logs', f"{model_name}-{int(time.time())}", '')
        tb_callback = TensorBoard(log_dir=log_dir)
        es_callback = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10, verbose=1, mode='min')
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1, mode='min')
        
        # ModelCheckpoint can be added if needed
        # model_checkpoint_path = os.path.join(self.DATA_PATH, f"{model_name}_best_during_train.h5") 
        # chkpt_callback = ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', verbose=1,
        #                                  save_best_only=True, save_weights_only=False, mode='min')
        # return [tb_callback, es_callback, lr_callback, chkpt_callback]
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

    def _build_attn_lstm_model(self, hidden_units=256):
        inputs_layer = Input(shape=(self.sequence_length_frames, self.num_input_values))
        lstm_out = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs_layer)
        attention_mul = self._attention_block(lstm_out, self.sequence_length_frames)
        attention_mul = Flatten()(attention_mul)
        x = Dense(2 * hidden_units, activation='relu')(attention_mul)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=[inputs_layer], outputs=x) # Corrected: use inputs_layer
        return model

    def build_and_compile_models(self, model_name, learning_rate=0.001):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if model_name == 'LSTM':
            print(f"[build_and_compile_models]: Building {model_name} model...")
            self.models[model_name] = self._build_lstm_model()
        elif model_name == 'LSTM_Attention_128HUs':
            print("[build_and_compile_models]: Building {model_name} model...")
            self.models['LSTM_Attention_128HUs'] = self._build_attn_lstm_model(hidden_units=256) 

        
        self.models[model_name].compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        print(f"[build_and_compile_models]: {model_name} model built and compiled.")
        self.models[model_name].summary(print_fn=lambda x: print(x) if "Total params" in x or "Trainable params" in x or "Layer (type)" in x else None)
    
    def train_model(self, model_name, epochs=50, batch_size=32):
        if model_name not in self.models:
            print(f"[train_model]: Error: Model {model_name} not found. Build models first.")
            return
        if self.X_train is None or self.y_train is None:
            print("[train_model]: Error: Training data not loaded/preprocessed.")
            return
        if self.X_train.shape[0] == 0:
            print("[train_model]: Error: Training data is empty.")
            return


        model = self.models[model_name]
        callbacks = self._get_training_callbacks(model_name)
        
        print(f"[train_model]: Training {model_name}...")
        history = model.fit(self.X_train, self.y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(self.X_val, self.y_val),
                            callbacks=callbacks,
                            verbose=1)
        print(f"{model_name} training finished.")
        return history

    def save_model(self, model_name, filename=None): # Renamed for clarity (saves full model)
        if model_name not in self.models:
            print(f"[save_model]: Model {model_name} not found.")
            return
        if filename is None:
            filename = f"{model_name}.h5"
        
        save_path = os.path.join(self.DATA_PATH, filename)
        self.models[model_name].save(save_path)
        print(f"[save_model]: Model {model_name} saved to {save_path}")

    def load_model(self, model_name, filename=None, compile_model=True): # Renamed for clarity
        if filename is None:
            filename = f"{model_name}.h5"
        load_path = os.path.join(self.DATA_PATH, filename)

        if not os.path.exists(load_path):
            print(f"[load_model]: Error: File {load_path} not found.")
            return False # Indicate failure

        try:
            loaded_model = tf.keras.models.load_model(load_path, compile=compile_model)
            self.models[model_name] = loaded_model
            print(f"[load_model]: Model {model_name} loaded from {load_path}")
            return True # Indicate success
        except Exception as e:
            print(f"[load_model]: Error loading model {model_name} from {load_path}: {e}")
            return False # Indicate failure

    def evaluate_all_models(self):
        if self.X_test is None or self.y_test is None or self.X_test.shape[0] == 0:
            print("Error: Test data not available or empty.")
            return

        self.eval_results = {
            'confusion_matrix': {}, 'accuracy': {}, 
            'precision': {}, 'recall': {}, 'f1_score': {}
        }
        y_true_labels = np.argmax(self.y_test, axis=1).tolist()

        for model_name, model_instance in self.models.items():
            if model_instance is None:
                print(f"Model {model_name} is not initialized. Skipping evaluation.")
                continue
            print(f"\nEvaluating {model_name}:")
            
            try:
                y_pred_probs = model_instance.predict(self.X_test, verbose=0)
            except Exception as e:
                print(f"Could not get predictions for {model_name}: {e}")
                continue

            y_pred_labels = np.argmax(y_pred_probs, axis=1).tolist()
            cm = multilabel_confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(self.num_classes)))
            self.eval_results['confusion_matrix'][model_name] = cm
            print(f"{model_name} Confusion Matrix:\n{cm}")

            acc = accuracy_score(y_true_labels, y_pred_labels)
            self.eval_results['accuracy'][model_name] = acc
            print(f"{model_name} Accuracy: {acc:.4f}")

            report = classification_report(y_true_labels, y_pred_labels, target_names=self.actions, 
                                           labels=list(range(self.num_classes)), output_dict=True, zero_division=0)
            
            self.eval_results['precision'][model_name] = report['weighted avg']['precision']
            self.eval_results['recall'][model_name] = report['weighted avg']['recall']
            self.eval_results['f1_score'][model_name] = report['weighted avg']['f1-score']
            
            print(f"{model_name} Weighted Avg Precision: {report['weighted avg']['precision']:.4f}")
            print(f"{model_name} Weighted Avg Recall: {report['weighted avg']['recall']:.4f}")
            print(f"{model_name} Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
            print(f"{model_name} Classification Report:\n{classification_report(y_true_labels, y_pred_labels, target_names=self.actions, labels=list(range(self.num_classes)), zero_division=0)}")
        
        return self.eval_results

    def _calculate_angle(self, a, b, c): # Using notebook's 2D angle calculation
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle 

    def _get_coordinates(self, landmark_list, side, joint_name):
        # landmark_list is results.pose_landmarks.landmark
        if not hasattr(self.mp_pose.PoseLandmark, f"{side.upper()}_{joint_name.upper()}"):
             raise ValueError(f"Invalid landmark name: {side.upper()}_{joint_name.upper()}")
        
        coord_idx = self.mp_pose.PoseLandmark[f"{side.upper()}_{joint_name.upper()}"].value
        lm = landmark_list[coord_idx]
        return [lm.x, lm.y] # Using 2D coordinates as in the notebook

    def _viz_joint_angle(self, image, angle, joint_coords_normalized, frame_width, frame_height):
        joint_pixel_coords = tuple(np.multiply(joint_coords_normalized, [frame_width, frame_height]).astype(int))
        cv2.putText(image, str(int(angle)), joint_pixel_coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def _count_reps_logic(self, image, current_action, landmark_list, frame_width, frame_height):
        # landmark_list is results.pose_landmarks.landmark
        if current_action == 'curl':
            shoulder = self._get_coordinates(landmark_list, 'left', 'shoulder')
            elbow = self._get_coordinates(landmark_list, 'left', 'elbow')
            wrist = self._get_coordinates(landmark_list, 'left', 'wrist')
            angle = self._calculate_angle(shoulder, elbow, wrist)

            if angle < 30: self.curl_stage = "up"
            if angle > 140 and self.curl_stage == 'up':
                self.curl_stage = "down"; self.counter[current_action] += 1
            self.press_stage, self.squat_stage = None, None
            self._viz_joint_angle(image, angle, elbow, frame_width, frame_height)

        elif current_action == 'press':
            shoulder = self._get_coordinates(landmark_list, 'left', 'shoulder')
            elbow = self._get_coordinates(landmark_list, 'left', 'elbow')
            wrist = self._get_coordinates(landmark_list, 'left', 'wrist')
            elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
            
            shoulder2elbow_dist = math.dist(shoulder, elbow) 
            shoulder2wrist_dist = math.dist(shoulder, wrist)

            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist): self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage == 'up'):
                self.press_stage = 'down'; self.press_counter += 1
            self.curl_stage, self.squat_stage = None, None
            self._viz_joint_angle(image, elbow_angle, elbow, frame_width, frame_height)

        elif current_action == 'squat':
            left_shoulder = self._get_coordinates(landmark_list, 'left', 'shoulder')
            left_hip = self._get_coordinates(landmark_list, 'left', 'hip')
            left_knee = self._get_coordinates(landmark_list, 'left', 'knee')
            left_ankle = self._get_coordinates(landmark_list, 'left', 'ankle')
            right_shoulder = self._get_coordinates(landmark_list, 'right', 'shoulder')
            right_hip = self._get_coordinates(landmark_list, 'right', 'hip')
            right_knee = self._get_coordinates(landmark_list, 'right', 'knee')
            right_ankle = self._get_coordinates(landmark_list, 'right', 'ankle')

            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
            
            thr = 165 
            if (left_knee_angle < thr and right_knee_angle < thr and 
                left_hip_angle < thr and right_hip_angle < thr): 
                self.squat_stage = "down"
            if (left_knee_angle > thr and right_knee_angle > thr and 
                left_hip_angle > thr and right_hip_angle > thr and self.squat_stage == 'down'):
                self.squat_stage = 'up'; self.squat_counter += 1
            self.curl_stage, self.press_stage = None, None
            
            self._viz_joint_angle(image, left_knee_angle, left_knee, frame_width, frame_height)
            self._viz_joint_angle(image, left_hip_angle, left_hip, frame_width, frame_height)
        else: 
            self.curl_stage, self.press_stage, self.squat_stage = None, None, None

    def _prob_viz(self, res_probs, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res_probs):
            action_name = self.actions[num]
            color_idx = num % len(self.colors)
            color = self.colors[color_idx]
            
            # Rectangle for probability bar
            bar_start_x = 0
            bar_start_y = 60 + num * 40
            bar_end_x = int(prob * 100) # Scale probability for bar width (e.g. prob * 100 pixels)
            bar_end_y = 90 + num * 40
            cv2.rectangle(output_frame, (bar_start_x, bar_start_y), (bar_end_x, bar_end_y), color, -1)
            
            # Text for action name
            text_x = 0
            text_y = 85 + num * 40
            cv2.putText(output_frame, action_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

    def run_real_time_inference(self, model_name_to_use, cap_source=0, threshold=0.5, 
                                save_video=False, output_video_filename=None, break_key='q'):
        if model_name_to_use not in self.models or self.models[model_name_to_use] is None:
            print(f"[run_real_time_inference]: Error: Model {model_name_to_use} not found or not loaded. Load or train it first.")
            if not self.load_model(model_name_to_use): # Try to load it
                print(f"[run_real_time_inference]: Failed to auto-load {model_name_to_use}. Please ensure it's available or train it.")
                return
        
        active_model = self.models[model_name_to_use]

        self.curl_counter, self.press_counter, self.squat_counter = 0, 0, 0
        self.curl_stage, self.press_stage, self.squat_stage = None, None, None
        sequence_data = []
        current_action_prediction = ''
        
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {cap_source}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        out_video = None
        if save_video:
            if output_video_filename is None:
                output_video_filename = f"{model_name_to_use}_real_time_test.avi"
            video_path = os.path.join(os.getcwd(), output_video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_video = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            print(f"Saving video to {video_path}")

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
            while cap.isOpened():
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

                    action_color = (50,50,50) # Default neutral color for top bar
                    if confidence >= threshold:
                        current_action_prediction = self.actions[predicted_action_idx]
                        color_idx = predicted_action_idx % len(self.colors)
                        action_color = self.colors[color_idx]
                    else:
                        current_action_prediction = ''
                    
                    cv2.rectangle(image, (0,0), (frame_width, 40), action_color, -1)
                    image = self._prob_viz(prediction_probs, image)
                    
                    if results.pose_landmarks and current_action_prediction:
                        try:
                            self._count_reps_logic(image, current_action_prediction, 
                                                  results.pose_landmarks.landmark, frame_width, frame_height)
                        except Exception as e:
                            print(f"Error in rep counting: {e}")
                
                # Display rep counts on the top bar
                cv2.putText(image, f'{self.actions[0][0]}: {self.curl_counter}', (15,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{self.actions[1][0]}: {self.press_counter}', (frame_width // 2 - 50, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{self.actions[2][0]}: {self.squat_counter}', (frame_width - 100, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                if save_video and out_video is not None: out_video.write(image)
                if cv2.waitKey(10) & 0xFF == ord(break_key): break
        
        cap.release()
        if out_video is not None: out_video.release()
        cv2.destroyAllWindows()
        print("Real-time inference finished.")

'''
if __name__ == '__main__':
    # This is a conceptual guide. Running all these steps takes time.
    # You'd typically run parts of this flow separately.

    decoder = ExerciseDecoder(
        actions=['curl', 'press', 'squat'], # Customize actions
        num_sequences_per_action=5, # Small number for quick test, original is 50
        sequence_length_frames=30,  # Standard 30 frames per sequence
        collection_start_folder=1   # Start folder for data collection
    )

    # --- OPTION 1: Collect Data, Train, and then Test ---
    # 1. Setup folders
    # decoder.setup_data_folders()

    # 2. Collect data
    # print("Please follow instructions on screen to collect data.")
    # decoder.collect_training_data() 
    # decoder.collection_start_folder += decoder.num_sequences_per_action # Prepare for next run

    # 3. Preprocess data
    # decoder.preprocess_data() 
    # if decoder.X_train is None or decoder.X_train.shape[0] == 0:
    #     print("Preprocessing failed or no data. Exiting.")
    #     exit()

    # 4. Build and compile models
    # decoder.build_and_compile_models(learning_rate=0.001)

    # 5. Train models
    # print("\nTraining LSTM model...")
    # decoder.train_model('LSTM', epochs=10, batch_size=16) # Small epochs for quick test
    # print("\nTraining Attention LSTM model...")
    # decoder.train_model('AttnLSTM', epochs=10, batch_size=16)

    # 6. Save models
    # decoder.save_model('LSTM')
    # decoder.save_model('AttnLSTM')
    
    # --- OPTION 2: Load Pre-trained Models and Test ---
    # Ensure model files (e.g., 'AttnLSTM_full_model.h5') exist in decoder.DATA_PATH
    print("Attempting to load pre-trained AttnLSTM model...")
    model_loaded = decoder.load_model('AttnLSTM') # Tries to load 'AttnLSTM_full_model.h5'

    if not model_loaded:
        print("Could not load AttnLSTM model. You might need to train it first (Option 1).")
        print("For demonstration, building and compiling a new (untrained) model.")
        decoder.build_and_compile_models() # Build untrained models if loading failed
        if 'AttnLSTM' not in decoder.models or decoder.models['AttnLSTM'] is None:
            print("Failed to prepare any model. Exiting.")
            exit()
        else:
            print("Proceeding with a new, untrained AttnLSTM model for structure testing.")
    
    # --- Common Steps After Model Preparation (Training or Loading) ---
    # 7. Evaluate models (optional, if you have test data from preprocessing)
    # if decoder.X_test is not None and decoder.X_test.shape[0] > 0:
    #     print("\nEvaluating models...")
    #     decoder.evaluate_all_models()
    # else:
    #     print("\nTest data not available for evaluation (e.g., if you skipped preprocessing).")

    # 8. Run real-time inference
    print("\nStarting real-time inference with AttnLSTM...")
    try:
        # Use the 'AttnLSTM' model (either trained or loaded)
        decoder.run_real_time_inference('AttnLSTM', threshold=0.7, save_video=True)
    except Exception as e:
        print(f"An error occurred during real-time inference: {e}")
        import traceback
        traceback.print_exc()

    print("\nExerciseDecoder example finished.")
'''
class FitnessApp(QWidget):
    def __init__(self):
        super().__init__()

        self.decoder = ExerciseDecoder(
            actions=['curl', 'press', 'squat'], # Customize actions
            num_sequences_per_action=20, # Small number for quick test, original is 50
            sequence_length_frames=30,  # Standard 30 frames per sequence
            collection_start_folder=1   # Start folder for data collection
        )
        self.init_ui()


        self.actions = []
        self.num_sequences_per_action = 50
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Smart Fitness Tracker")
        self.setGeometry(100, 100, 450, 350)

        # 標題
        self.title = QLabel("🏋️‍♀️ Smart Fitness App", self)
        self.title.setFont(QFont("Arial", 28, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        # 輸入欄：Actions
        self.actions_input = QLineEdit(self)
        self.actions_input.setPlaceholderText("Enter actions (e.g., curl,press,squat)")
        self.actions_input.textChanged.connect(self.update_actions)

        # 輸入欄：Num Sequences
        self.num_seq_label = QLabel("Sequences per action:")
        self.num_seq_spinbox = QSpinBox(self)
        self.num_seq_spinbox.setRange(1, 1000)
        self.num_seq_spinbox.setValue(50)
        self.num_seq_spinbox.valueChanged.connect(self.update_num_sequences)

        # 按鈕
        self.warmup_btn = QPushButton("Warm Up")
        self.finetune_btn = QPushButton("Fine Tune")
        self.start_btn = QPushButton("Start")

        self.warmup_btn.clicked.connect(self.warmup_action)
        self.finetune_btn.clicked.connect(self.finetune_action)
        self.start_btn.clicked.connect(self.start_action)

        # 佈局設計
        layout = QVBoxLayout()
        layout.addWidget(self.title)

        layout.addWidget(QLabel("Actions (comma-separated):"))
        layout.addWidget(self.actions_input)

        seq_layout = QHBoxLayout()
        seq_layout.addWidget(self.num_seq_label)
        seq_layout.addWidget(self.num_seq_spinbox)
        layout.addLayout(seq_layout)

        layout.addWidget(self.warmup_btn)
        layout.addWidget(self.finetune_btn)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)
        self.setStyleSheet(self.load_styles())

    def load_styles(self):
        return """
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QLineEdit {
                border: 2px solid #5F9EA0;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                background-color: #f5f5f5;
                color: #333;
            }

            QSpinBox {
                border: 2px solid #5F9EA0;
                border-radius: 5px;
                padding: 2px 5px;
                font-size: 14px;
                background-color: #f5f5f5;
                color: #333;
            }

            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #5F9EA0;
                background-color: #0;
            }

            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #5F9EA0;
                background-color: #0;
            }

            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 10px;
                height: 10px;
                
            }

            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                margin-top: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

    def update_actions(self, text):
        self.actions = [a.strip() for a in text.split(',') if a.strip()]
        print(self.actions)
    def update_num_sequences(self, value):
        self.num_sequences_per_action = value
        print(self.num_sequences_per_action)
    def warmup_action(self):
        if not self.actions:
            QMessageBox.warning(self, "Input Error", "Please enter at least one action.")
            return

        QMessageBox.information(
            self,
            "Warm Up",
            f"Actions: {', '.join(self.actions)}\nSequences per action: {self.num_sequences_per_action}"
        )
        print("Warm up started...")
        self.decoder = ExerciseDecoder(
            actions=self.actions, # Customize actions
            num_sequences_per_action=self.num_sequences_per_action, # Small number for quick test, original is 50
            sequence_length_frames=30,  # Standard 30 frames per sequence
            collection_start_folder=1   # Start folder for data collection
        )
    def finetune_action(self):
        print("Fine tuning...")
        # --- OPTION 1: Collect Data, Train, and then Test ---
        # 1. Setup folders
        self.decoder.setup_data_folders()

        # 2. Collect data
        print("[finetune_action]: Please follow instructions on screen to collect data.")
        self.decoder.collect_training_data() 
        self.decoder.collection_start_folder += self.decoder.num_sequences_per_action # Prepare for next run

        # 3. Preprocess data
        self.decoder.preprocess_data() 
        if self.decoder.X_train is None or self.decoder.X_train.shape[0] == 0:
           print("[finetune_action]: Preprocessing failed or no data. Exiting.")
           exit()

        # 4. Build and compile models
        self.decoder.build_and_compile_models(model_name='LSTM', learning_rate=0.001)
        self.decoder.build_and_compile_models(model_name='LSTM_Attention_128HUs', learning_rate=0.001)

        # 5. Train models
        print("\n[finetune_action]: Training LSTM model...")
        self.decoder.train_model('LSTM', epochs=10, batch_size=16) # Small epochs for quick test
        print("\n[finetune_action]: Training Attention LSTM model...")
        self.decoder.train_model('LSTM_Attention_128HUs', epochs=10, batch_size=16)

        # 6. Save models
        self.decoder.save_model('LSTM')
        self.decoder.save_model('LSTM_Attention_128HUs')

        print('[finetune_action]: END')

    def start_action(self):

        weight_name = 'LSTM_Attention_128HUs'
        print("[start_action]: Starting workout session...")
        print("[start_action]: Attempting to load pre-trained {weight_name} model...".format(weight_name = weight_name))
        
        model_loaded = self.decoder.load_model(weight_name) # Tries to load 'AttnLSTM_full_model.h5'

        if not model_loaded:
            print(f"[start_action]: Could not load {weight_name} model. You might need to train it first (Option 1).")
            print("[start_action]: For demonstration, building and compiling a new (untrained) model.")
            self.decoder.build_and_compile_models() # Build untrained models if loading failed
            if weight_name not in self.decoder.models or self.decoder.models[weight_name] is None:
                print("[start_action]: Failed to prepare any model. Exiting.")
                exit()
            else:
                print("[start_action]: Proceeding with a new, untrained {weight_name} model for structure testing.")
        
        # --- Common Steps After Model Preparation (Training or Loading) ---
        # 7. Evaluate models (optional, if you have test data from preprocessing)
        # if decoder.X_test is not None and decoder.X_test.shape[0] > 0:
        #     print("\nEvaluating models...")
        #     decoder.evaluate_all_models()
        # else:
        #     print("\nTest data not available for evaluation (e.g., if you skipped preprocessing).")

        # 8. Run real-time inference
        print("\n[start_action]: Starting real-time inference with AttnLSTM...")
        try:
            # Use the 'AttnLSTM' model (either trained or loaded)
            self.decoder.run_real_time_inference(weight_name, threshold=0.7, save_video=True)
        except Exception as e:
            print(f"[start_action]: An error occurred during real-time inference: {e}")
            import traceback
            traceback.print_exc()

        print("\n[start_action]: ExerciseDecoder example finished.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FitnessApp()
    window.show()
    sys.exit(app.exec_())
