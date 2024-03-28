import cv2 as cv
import numpy as np

img = cv.imread("Photos/cats.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blank = np.zeros(img.shape, dtype='uint8')

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny edge detection", canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Threshold image", thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("Drawn contours", blank)

cv.waitKey(0)
