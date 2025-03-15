import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Constants for EAR and MAR (Mouth Aspect Ratio)
EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold for blink
MAR_THRESHOLD = 0.5   # Mouth Aspect Ratio threshold for yawn
BLINK_CONSEC_FRAMES = 3  # Number of consecutive frames to count as a blink
YAWN_CONSEC_FRAMES = 10  # Number of consecutive frames to count as a yawn

# Status variables
blink_counter = 0  # Total number of blinks
drowsy_duration = 0  # Duration of drowsiness in frames
yawn_counter = 0  # Total number of yawns
active_duration = 0  # Duration of active state in frames
status = ""
color = (0, 0, 0)

# State variables
eyes_closed = False  # Track if eyes are currently closed
mouth_open = False   # Track if mouth is currently open

# Moving average for MAR stabilization
MAR_HISTORY = deque(maxlen=5)  # Store last 5 MAR values
mar_avg = 0.0

# FPS variables
frame_count = 0
start_time = time.time()
fps = 0

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    # Vertical distances
    A = dist.euclidean(mouth[2], mouth[10])  # Top to bottom
    B = dist.euclidean(mouth[4], mouth[8])   # Inner top to inner bottom
    # Horizontal distance
    C = dist.euclidean(mouth[0], mouth[6])   # Left to right
    # MAR calculation
    mar = (A + B) / (2.0 * C)
    return mar

# Main loop
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    # Detect faces
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Extract eye and mouth landmarks
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for mouth
        mar = mouth_aspect_ratio(mouth)

        # Stabilize MAR using moving average
        MAR_HISTORY.append(mar)
        mar_avg = np.mean(MAR_HISTORY)

        # Check for blinks
        if ear < EAR_THRESHOLD:
            if not eyes_closed:
                eyes_closed = True
                blink_counter += 1  # Increment blink counter only once per blink
        else:
            eyes_closed = False

        # Check for yawns
        if mar_avg > MAR_THRESHOLD:
            if not mouth_open:
                mouth_open = True
                yawn_counter += 1  # Increment yawn counter only once per yawn
        else:
            mouth_open = False

        # Determine drowsiness and sleep status
        if ear < EAR_THRESHOLD:
            drowsy_duration += 1
            if drowsy_duration > 6:
                status = "Sleeping !!!"
                color = (255, 0, 0)
        elif ear < 0.25:  # Drowsy threshold
            drowsy_duration += 1
            if drowsy_duration > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
        else:
            drowsy_duration = 0
            active_duration += 1
            if active_duration > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display EAR and MAR values
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display blink and yawn counters
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn_counter}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display status
        cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()