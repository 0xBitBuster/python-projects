import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Resources/Photos/park.jpg")
cv.imshow("original", img)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# BGR to l+a+b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Show matplotlib graph
plt.imshow(rgb)
plt.show()

cv.imshow("edited", rgb)
cv.waitKey(0)
