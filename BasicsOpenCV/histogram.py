import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Resources/Photos/cats.jpg")
cv.imshow("original", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Grayscale histogram
# gray_hist = cv.calcHist([gray], [0], None, [256],[0, 256])
#
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

# Color histogram
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
