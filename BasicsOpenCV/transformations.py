import cv2 as cv
import numpy as np

img = cv.imread("Photos/park.jpg")
cv.imshow("img", img)


# Move image
def translate(image, x, y):
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], image.shape[0])

    return cv.warpAffine(image, translation_matrix, dimensions)


translated = translate(img, -50, 50)
cv.imshow("translated", translated)


# Rotate image
def rotate(image, angle, rot_point=None):
    (height, width) = image.shape[:2]

    if rot_point is None:
        rot_point = (width // 2, height // 2)

    rot_matrix = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(image, rot_matrix, dimensions)


rotated = rotate(img, 45)
cv.imshow("rotated", rotated)


# Resizing (For downscaling INTER_AREA, for up-scaling INTER_LINEAR (faster) or INTER_CUBIC (slower, higher quality)
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("resized", resized)


# Flipping (0 = vertically, 1 = horizontally, -1 = both)
flip = cv.flip(img, 0)
cv.imshow("flipped", flip)


# Cropping
cropped = img[50:200, 200: 400]
cv.imshow("cropped", cropped)


cv.waitKey(0)
