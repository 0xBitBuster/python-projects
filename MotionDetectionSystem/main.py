import threading
import winsound
import cv2 as cv
import imutils

# Initialize video capture
capture = cv.VideoCapture(0, cv.CAP_DSHOW)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Capture and process the first frame
ret, start_frame = capture.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv.cvtColor(start_frame, cv.COLOR_BGR2GRAY)
start_frame = cv.GaussianBlur(start_frame, (21, 21), 0)

# Initialize alarm variables
alarm = False
alarm_mode = False
alarm_counter = 0
sensitivity = 300


def alarm_beep():
    global alarm

    for _ in range(3):
        if not alarm_mode:
            break
        print("ALARM!")
        winsound.Beep(2500, 1000)
    alarm = False


while True:
    ret, frame = capture.read()
    frame = imutils.resize(frame, width=500)

    if alarm_mode:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray, (5, 5), 0)

        difference = cv.absdiff(start_frame, frame_gray)
        threshold = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)[1]
        start_frame = frame_gray

        # Increment alarm counter if movement detected
        if threshold.sum() > sensitivity:
            alarm_counter += 1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1

        cv.imshow("Camera", threshold)
    else:
        cv.imshow("Camera", frame)

    # Trigger alarm if movement detected continuously
    if alarm_counter > 20:
        if not alarm:
            alarm = True
            threading.Thread(target=alarm_beep).start()

    # Detect key press
    key_pressed = cv.waitKey(20)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    elif key_pressed == ord("q"):
        alarm_mode = False
        break

capture.release()
cv.destroyAllWindows()
