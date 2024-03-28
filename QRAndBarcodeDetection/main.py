import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

capture = cv.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

while True:
    success, img = capture.read()

    for barcode in decode(img):
        data = barcode.data.decode('utf-8')
        print(data)

        # Draw a polygon around the barcode using the points
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img, [pts], True, (255, 0, 255), 5)

        # Draw the decoded data as text near the barcode
        pts2 = barcode.rect
        cv.putText(img, data, (pts2[0], pts2[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv.imshow('Result', img)
    if cv.waitKey(20) == ord("e"):
        break

capture.release()
cv.destroyAllWindows()
