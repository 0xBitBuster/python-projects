import cv2 as cv
import numpy as np
import pytesseract
from imutils.perspective import four_point_transform

capture = cv.VideoCapture(0, cv.CAP_DSHOW)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

count = 0
document_contour = np.array([])
FONT = cv.FONT_HERSHEY_SIMPLEX
WIDTH, HEIGHT = 800, 600
capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def image_processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

    return threshold


def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 1000:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)


def center_text(image, text):
    text_size = cv.getTextSize(text, FONT, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv.putText(image, text, (text_x, text_y), FONT, 2, (255, 0, 255), 5, cv.LINE_AA)


while True:
    _, frame = capture.read()
    frame = cv.rotate(frame, cv.ROTATE_180)
    frame_copy = frame.copy()

    scan_detection(frame_copy)
    cv.imshow("Input", frame)

    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    cv.imshow("Warped", frame)

    processed = image_processing(warped)
    processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
    cv.imshow("Processed", frame)

    pressed_key = cv.waitKey(20) & 0xFF

    if pressed_key == 27:
        break

    elif pressed_key == ord('s'):
        cv.imwrite("output/scanned_" + str(count) + ".jpg", processed)
        count += 1

        center_text(frame, "Scan saved")
        cv.imshow("Input", frame)
        cv.waitKey(500)

    elif pressed_key == ord('o'):
        file = open("output/recognized_" + str(count - 1) + ".txt", "w")
        ocr_text = pytesseract.image_to_string(warped) # OPTIONAL!!!????

        file.write(ocr_text)
        file.close()

        center_text(frame, "Text saved")
        cv.imshow("Input", frame)
        cv.waitKey(500)


cv.destroyAllWindows()
