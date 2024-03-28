import cv2 as cv
import numpy as np


# 1. Initialize variables
#                  height / width / color channel
image = np.zeros((500, 500, 3), dtype='uint8')


# 1. Draw rectangle using numpy
#    y1, y2 | x1, x2     B  G  R
image[10:50, 10:50] = [0, 0, 255]


# 2. Draw rectangle using cv
cv.rectangle(image, (70, 10), (110, 50), (0, 255, 0), thickness=cv.FILLED)
cv.rectangle(image, (130, 10), (170, 50), (0, 255, 0), thickness=2)


# 3. Draw a circle using cv
cv.circle(image, (image.shape[1] // 2, image.shape[0] // 2), 30, (0, 255, 0), thickness=2)


# 4. Draw a line
cv.line(image, (190, 10), (230, 50), (255, 255, 255), thickness=2)


# 5. Write text
cv.putText(image, "Hello", (image.shape[1] // 2, image.shape[0] - 50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

cv.imshow("Image", image)
cv.waitKey(0)
