import cv2 as cv

img = cv.imread("Resources/Photos/lady.jpg")
cv.imshow("original", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f"Detected {len(faces_rect)} faces!")

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)
