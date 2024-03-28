import cv2 as cv

img = cv.imread("Resources/Photos/park.jpg")
cv.imshow("original", img)

# Averaging (Calculates average of surrounding pixels)
avg = cv.blur(img, (7, 7))
cv.imshow('Average Blur', avg)

# Gaussian Blur (Calculates average of surrounding pixels with weights)
gauss = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow("Gaussian Blur", gauss)

# Median Blur (Calculates median of surrounding pixels)
med = cv.medianBlur(img, 7)
cv.imshow("Median Blur", med)

# Bilateral Blur (Preserves edges while smoothing)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow("Bilateral Blur", bilateral)

cv.waitKey(0)
