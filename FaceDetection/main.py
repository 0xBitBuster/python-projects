import cv2 as cv
import cvzone
from ultralytics import YOLO

face_model = YOLO('models/yolov8n-face.pt')
capture = cv.VideoCapture('videos/walking.mp4')

while True:
    _, frame = capture.read()

    face_result = face_model.predict(frame, conf=0.40)
    for info in face_result:
        params = info.boxes
        for box in params:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    cv.imshow("Video", frame)

    if cv.waitKey(20) == ord("e"):
        break

capture.release()
cv.destroyAllWindows()
