import cv2
import mediapipe as mp
import numpy as np
import joblib

# Initialize mediapipe Hands Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the SVM model and scaler
model_filename = "svm_model.pkl"
clf = joblib.load(model_filename)
scaler_filename = "scaler.pkl"
scaler = joblib.load(scaler_filename)

# Load labels
label_file = "labels.txt"
with open(label_file, 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

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
        distances.append(distance/reference_distance)  # Normalize the distance using the reference distance

    return distances

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 加入鏡像
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            # Distinguish between left and right hand
            hand_label = "Right" if results.multi_handedness[index].classification[0].label == "Left" else "Left"

            distances = compute_distances(landmarks)
            distances = scaler.transform([distances])

            prediction = clf.predict(distances)
            confidence = np.max(clf.predict_proba(distances))

            label = labels[prediction[0]]
            display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"

            cv2.putText(frame, display_text, (10, 30 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # To visualize the landmarks of the hand
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Demo', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()