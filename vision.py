import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from playsound import playsound



# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
blink_count = 0
yawn_count = 0
blink_threshold = 0.21
yawn_threshold = 20

# Variables for blink and yawn detection
eye_closed = False
blink_detected = False
mouth_open = False
yawn_detected = False

#to play sound when drowsy and sleepy
sound="./alert.wav"

def play_sound():
    playsound(sound)

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > blink_threshold and ratio <= 0.25:
        return 1
    else:
        return 0

def detect_yawn(landmarks):
    top_lip = landmarks[50:53]
    top_lip = np.concatenate((top_lip, landmarks[61:64]))

    low_lip = landmarks[56:59]
    low_lip = np.concatenate((low_lip, landmarks[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Eye blink detection
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            if not eye_closed:
                eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
                eye_closed = False
                blink_detected = True

        # Yawn detection
        yawn_distance = detect_yawn(landmarks)
        if yawn_distance > yawn_threshold:
            if not mouth_open:
                mouth_open = True
        else:
            if mouth_open:
                yawn_count += 1
                mouth_open = False

        # Status update
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING "
                color = (255, 0, 0)
                play_sound()
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy "
                color = (0, 0, 255)
                play_sound()
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active "
                color = (0, 255, 0)

        # Display status and counts
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()