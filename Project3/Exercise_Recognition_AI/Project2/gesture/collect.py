import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def compute_distances(landmarks):
    distances = []
    
    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    reference_pair = (0, 9)
    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)
    
    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    return distances

# Ask user for filename
filename = input("Please enter the filename for data: ")
save_path = "dataset/" + filename

# Check if the 'dataset' directory exists, if not, create it
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cap = cv2.VideoCapture(0)
data_collection = []

collecting = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # 加入鏡像
    frame = cv2.flip(frame, 1)

    if not collecting:
        cv2.putText(frame, "Press SPACE to start data collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        elapsed_time = int(time.time() - start_time)
        remaining_time = 10 - elapsed_time
        cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if elapsed_time >= 10:
            break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and collecting:
        for landmarks in results.multi_hand_landmarks:
            distances = compute_distances(landmarks)
            data_collection.append(distances)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)
    
    if key == 32 and not collecting:
        collecting = True
        start_time = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert the data_collection list to numpy array and save
np.save(save_path, np.array(data_collection))
print(f"Data saved to {save_path}")