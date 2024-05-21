import os
import cv2 as cv

sample = cv.imread('datasets/SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_CR.BMP')

best_score = 0
filename = None
img = None
kp1, kp2, mp = None, None, None

for file in [file for file in os.listdir('datasets/SOCOFing/Real')][:1000]:
    fingerprint_image = cv.imread('datasets/SOCOFing/Real/' + file)
    sift = cv.SIFT().create()

    key_points_1, descriptors_1 = sift.detectAndCompute(sample, None)
    key_points_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]
    key_points = min([key_points_1, key_points_2], key=len)

    if (len(match_points) / len(key_points) * 100) > best_score:
        best_score = (len(match_points) / len(key_points) * 100)
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = key_points_1, key_points_2, match_points

if filename is None:
    print("NO MATCH WAS FOUND!")
    exit(0)

print("BEST MATCH: " + filename)
print("SCORE: " + str(best_score))

result = cv.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv.resize(result, None, fx=4, fy=4)
cv.imshow("Fingerprint Result", result)
cv.waitKey(0)
cv.destroyAllWindows()
