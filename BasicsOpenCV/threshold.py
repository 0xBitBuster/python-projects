import cv2 as cv

img = cv.imread("Resources/Photos/cats.jpg")
cv.imshow("original", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Simple Thresholding (Threshold is not calculated, but global)
threshold, thresh = cv.threshold(gray, 155, 255, cv.THRESH_BINARY)
cv.imshow("Simple Thresholding", thresh)

# Simple Inverse Thresholding
threshold_inv, thresh_inv = cv.threshold(gray, 155, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Inverse Thresholding", thresh_inv)

# Adaptive Thresholding (Threshold is calculated for smaller regions)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 8)
cv.imshow("Adaptive Thresholding", adaptive_thresh)

cv.waitKey(0)
