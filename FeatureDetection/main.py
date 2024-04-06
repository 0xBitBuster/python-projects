import cv2 as cv
import os

# Initialize video capture, ORB and BFMatcher
capture = cv.VideoCapture(0)
orb = cv.ORB().create()
bf = cv.BFMatcher()

# Load base images
base_images_path = "images"
base_images_names = os.listdir(base_images_path)
base_images = []
for image_name in base_images_names:
    image_path = os.path.join(base_images_path, image_name)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    base_images.append(image)


# Function to find descriptors of an image
def find_descriptors(images):
    des_list = []
    for base_image in images:
        kp, des = orb.detectAndCompute(base_image, None)
        des_list.append(des)

    return des_list


# Function to find the ID of the matched image
def find_matching_image_id(img, descriptor_list, thresh=20):
    kp2, des2 = orb.detectAndCompute(img, None)
    match_list = []
    final_index = -1

    for des in descriptor_list:
        matches = bf.knnMatch(des, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        match_list.append(len(good_matches))

    if len(match_list) != 0:
        most_matches = max(match_list)
        if most_matches > thresh:
            final_index = match_list.index(most_matches)

    return final_index


descriptors = find_descriptors(base_images)

while True:
    ret, frame = capture.read()
    frame_original = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    class_id = find_matching_image_id(frame, descriptors)
    if class_id != -1:
        cv.putText(frame_original, base_images_names[class_id], (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv.imshow("Video Capture", frame_original)
    if cv.waitKey(20) == ord("e"):
        break

capture.release()
cv.destroyAllWindows()
