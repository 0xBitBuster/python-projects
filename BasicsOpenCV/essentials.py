import cv2 as cv

img = cv.imread("Photos/park.jpg")
cv.imshow("img", img)

# Grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# Blur
# blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow("Blur", blur)

# Edge Cascade
# canny = cv.Canny(img, 125, 175)  # strong edges, weak edges
# cv.imshow("Canny edge detection", canny)

# Dilating the image
# dilated = cv.dilate(canny, (3, 3), iterations=1)
# cv.imshow("Dilated", dilated)

# Eroding
# eroded = cv.erode(dilated, (3, 3), iterations=1)
# cv.imshow("Eroded", eroded)

# Resize
# resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
# cv.imshow("Resized", resized)

# Cropping
cropped = img[50:200, 200: 400]
cv.imshow("cropped", cropped)

cv.waitKey(0)
