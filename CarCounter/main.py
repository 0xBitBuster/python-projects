import numpy as np
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

capture = cv.VideoCapture("assets/videos/cars.mp4")
model = YOLO("models/yolov8n.pt")

all_classes = model.names
interesting_classes = ["car", "motorbike", "bus", "train", "truck"]

mask = cv.imread("assets/images/mask.jpg")
counter_background = cv.imread("assets/images/counter_background.png", cv.IMREAD_UNCHANGED)

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
limits = [423, 297, 673, 297]
total_count = []

while True:
    ret, frame = capture.read()
    if not ret:
        # If video is over, reset the video capture to the beginning
        capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_region = cv.bitwise_and(frame, mask)
    frame = cvzone.overlayPNG(frame, counter_background, (0, 0))
    results = model(frame_region, stream=True, classes=[2, 3, 5, 6, 7])
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence & Class name
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = all_classes[cls]

            # Check if detection is interesting
            if current_class in interesting_classes and conf > 0.3:
                cur_arr = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, cur_arr])

    # Update tracker
    tracker_results = tracker.update(detections)
    cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    for result in tracker_results:
        x1, y1, x2, y2, t_id = result
        x1, y1, x2, y2, t_id = int(x1), int(y1), int(x2), int(y2), int(t_id)
        w, h = x2 - x1, y2 - y1

        # Draw a rectangle around the car
        cvzone.cornerRect(frame, (x1, y1, w, h), rt=2, l=10, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'{t_id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Draw a circle at car center
        center_x, center_y = x1 + w // 2, y1 + h // 2
        cv.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)

        # Check if car center point is within the counter line
        if limits[0] < center_x < limits[2] and limits[1] - 15 < center_y < limits[1] + 15:
            if total_count.count(t_id) == 0:
                total_count.append(t_id)
                cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

    # Display count
    cv.putText(frame, str(len(total_count)), (210, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 8)

    # Display frame
    cv.imshow("Car Counter", frame)
    if cv.waitKey(1) == ord("e"):
        break

capture.release()
cv.destroyAllWindows()
