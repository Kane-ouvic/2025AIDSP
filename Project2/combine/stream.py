import cv2
import sys

cap = cv2.VideoCapture("rtmp://140.116.56.6:1935/live")
if not cap.isOpened():
    print("Error: Could not open video source.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



