import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)


# Change Resolution from Live Videos
def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)


changeRes(200, 150)


# Resize frame from all media
def resize_frame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)


while True:
    isOk, frame = capture.read()

    canny = cv.Canny(frame, 125, 175)
    cv.imshow("Video", canny)

    if cv.waitKey(20) == ord("e"):
        break


capture.release()
cv.destroyAllWindows()
