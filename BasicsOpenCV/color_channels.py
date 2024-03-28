import cv2 as cv
import numpy as np

img = cv.imread("Resources/Photos/park.jpg")
cv.imshow("original", img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Get back split color channel grayscale images with density
b, g, r = cv.split(img)

# Merge color channels together
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

merged = cv.merge([b, g, r])
cv.imshow("merged", merged)

cv.waitKey(0)
