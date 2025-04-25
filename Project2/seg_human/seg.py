import cv2
import mediapipe as mp
import numpy as np

# 初始化 mediapipe selfie segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 啟動 webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)

    # 將影像轉為 RGB，符合 mediapipe 格式
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)

    # 獲取 segmentation mask（值為 [0,1]）
    mask = results.segmentation_mask

    # 將 mask threshold，建立 binary mask
    condition = mask > 0.5
    condition = condition.astype(np.uint8) * 255

    # 將 binary mask 轉為三通道（用來展示或分割）
    condition_3ch = cv2.merge([condition, condition, condition])

    # 前景為原始 frame，背景填黑（或你可改成透明、其他圖片等）
    foreground = cv2.bitwise_and(frame, condition_3ch)

    # 顯示結果
    cv2.imshow('Real-time Human Segmentation', foreground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
