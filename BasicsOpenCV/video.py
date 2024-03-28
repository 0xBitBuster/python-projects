import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    isOk, frame = capture.read()

    cv.imshow("Video", frame)

    if cv.waitKey(20) == ord("e"):
        break

capture.release()
cv.destroyAllWindows()