import cv2 as cv
from ultralytics import YOLO
from utils import *


def process_detections(frame, model):
    predictions_parsed = {}
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]

            # Calculate the center and dimensions of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            new_bbox = {
                'x': center_x,
                'y': center_y,
                'width': width,
                'height': height
            }

            if cls_name not in predictions_parsed:
                predictions_parsed[cls_name] = {
                    'bbox': new_bbox
                }
            else:
                existing_bbox = predictions_parsed[cls_name]['bbox']
                merged_bbox = merge_bboxes(existing_bbox, new_bbox)
                predictions_parsed[cls_name]['bbox'] = merged_bbox

    return predictions_parsed


def draw_bboxes_and_count(frame, predictions_parsed):
    detected_cards = []
    for cls_name, bbox_info in predictions_parsed.items():
        bbox = bbox_info['bbox']
        x1 = int(bbox['x'] - bbox['width'] / 2)
        y1 = int(bbox['y'] - bbox['height'] / 2)
        x2 = int(bbox['x'] + bbox['width'] / 2)
        y2 = int(bbox['y'] + bbox['height'] / 2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        detected_cards.append(cls_name)

    # Calculate the best possible count for the detected cards
    count = calculate_best_count(detected_cards)

    # Draw the count on the frame
    count_text = f"Count: {count}"
    cv.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)
    cv.putText(frame, count_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


# Load the model
model = YOLO("yolov8s_playing_cards.pt")
cap = cv.VideoCapture(0)
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Only perform detection on every 5th frame
    if frame_counter % 5 == 0:
        predictions_parsed = process_detections(frame, model)
        draw_bboxes_and_count(frame, predictions_parsed)
        cv.imshow("Detections", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

    frame_counter += 1

cap.release()
cv.destroyAllWindows()
