import cv2 as cv
import pickle
import cvzone
import numpy as np

PARKING_LOT_WIDTH, PARKING_LOT_HEIGHT = 107, 48

with open('car_park_positions', 'rb') as f:
    position_list = pickle.load(f)

capture = cv.VideoCapture("car_park.mp4")


def check_parking_space(frame, dilated):
    space_counter = 0

    for position in position_list:
        x, y = position
        car_crop = dilated[y:y+PARKING_LOT_HEIGHT, x:x+PARKING_LOT_WIDTH]
        pixel_count = cv.countNonZero(car_crop)

        # If pixel count is less than 900 = parking lot is empty
        if pixel_count < 900:
            space_counter += 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv.rectangle(frame, position, (x + PARKING_LOT_WIDTH, y + PARKING_LOT_HEIGHT), color, 2)
        cvzone.putTextRect(frame, str(pixel_count), (x, y+PARKING_LOT_HEIGHT-3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(frame, f'Free: {space_counter}/{len(position_list)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

    return frame


while True:
    # Start video from beginning if it is the last frame
    if capture.get(cv.CAP_PROP_POS_FRAMES) == capture.get(cv.CAP_PROP_FRAME_COUNT):
        capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_blur = cv.GaussianBlur(frame_gray, (3, 3), 1)
    frame_thresh = cv.adaptiveThreshold(frame_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 16)
    frame_median = cv.medianBlur(frame_thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    frame_dilate = cv.dilate(frame_median, kernel, iterations=1)

    frame = check_parking_space(frame, frame_dilate)

    cv.imshow("Car Park", frame)
    cv.waitKey(10)
