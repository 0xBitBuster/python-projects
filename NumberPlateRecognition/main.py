import cv2 as cv
from ultralytics import YOLO

from sort import *
from utils import get_car, read_license_plate, write_csv

# Define variables
results = {}
mot_tracker = Sort()
vehicles = [2, 3, 5, 6, 7]  # Vehicle class id's of model (e.g. 2=car, 3=motorbike)

# Load models
coco_model = YOLO('models/yolov8n.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

# Load video
capture = cv.VideoCapture("videos/sample.mp4")

# Read frames
frame_count = 0
while True:
    frame_count += 1
    ret, frame = capture.read()
    if not ret:
        break

    results[frame_count] = {}

    frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect vehicles
    detections = coco_model(frame, verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame, verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        frame = cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Assign license plate to car
        x1_car, y1_car, x2_car, y2_car, car_id = get_car(license_plate, track_ids)

        if car_id == -1:
            continue

        # Crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Process license plate
        license_plate_crop_gray = cv.cvtColor(license_plate_crop, cv.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv.threshold(license_plate_crop_gray, 64, 255, cv.THRESH_BINARY_INV)

        # Read license plate number
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

        # Write to results
        if license_plate_text is not None:
            results[frame_count][car_id] = {
                'car': {
                    'bbox': [x1_car, y1_car, x2_car, y2_car]
                },
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score
                }
            }

# Write results
write_csv(results, './test.csv')
