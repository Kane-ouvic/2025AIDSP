import cv2
from ultralytics import YOLO

# 載入 YOLOv8 預訓練模型（預設包含 person 類別）
model = YOLO('yolov8n.pt')  # 你也可以改成 yolov8s.pt, yolov5s.pt...

# 開啟 webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 鏡像顯示
    frame = cv2.flip(frame, 1)

    # 執行 YOLO 推論（使用 BGR frame）
    results = model(frame, stream=True)

    person_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 顯示人數
    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow('YOLOv8 Real-Time People Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
