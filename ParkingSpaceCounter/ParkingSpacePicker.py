import cv2 as cv
import pickle

PARKING_LOT_WIDTH, PARKING_LOT_HEIGHT = 107, 48

try:
    with open('car_park_positions', 'rb') as f:
        position_list = pickle.load(f)
except:
    position_list = []


def mouse_click(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        position_list.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        for i, position in enumerate(position_list):
            x1, y1 = position
            if x1 < x < x1 + PARKING_LOT_WIDTH and y1 < y < y1 + PARKING_LOT_HEIGHT:
                position_list.pop(i)

    with open('car_park_positions', 'wb') as f:
        pickle.dump(position_list, f)


while True:
    img = cv.imread('car_park.png')

    for pos in position_list:
        cv.rectangle(img, pos, (pos[0] + PARKING_LOT_WIDTH, pos[1] + PARKING_LOT_HEIGHT), (255, 0, 255), 2)

    cv.imshow("Car Park", img)
    cv.setMouseCallback("Car Park", mouse_click)
    cv.waitKey(1)
