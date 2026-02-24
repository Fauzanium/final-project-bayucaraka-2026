#make a program that uses the best.pt model to detect objects using webcam and display the results in real-time. The program should also display the FPS (frames per second) on the screen.
import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("./vision/runs/detect/train2/weights/best.pt").to("cpu")
cap = cv2.VideoCapture(1)
fps = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
