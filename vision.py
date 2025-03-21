import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import time
import pygame  # Using pygame instead of playsound for non-blocking audio

# Initialize pygame for audio
pygame.mixer.init()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status variables
sleep_frames = 0
drowsy_frames = 0
active_frames = 0
status = "No Face Detected"
color = (200, 200, 200)  # Default gray

# Detection counters
blink_count = 0
yawn_count = 0

# Detection state variables
eye_closed = False
consecutive_closed = 0
mouth_open = False
consecutive_yawn = 0

# Alert cooldown to prevent continuous alerts
last_alert_time = 0
ALERT_COOLDOWN = 3.0  # seconds

# Detection thresholds (can be adjusted)
EYE_AR_THRESHOLD = 0.25  # Eye aspect ratio threshold for blink detection
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for blink detection
YAWN_THRESHOLD = 20  # Mouth aspect ratio threshold for yawn detection
DROWSY_THRESHOLD = 4  # Consecutive frames to determine drowsiness
SLEEP_THRESHOLD = 6  # Consecutive frames to determine sleeping

# Create a simple beep sound for alerts
sample_rate = 44100
duration = 0.5  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), False)
# Generate a 440 Hz sine wave
note = np.sin(2 * np.pi * 440 * t)
# Make it stereo by duplicating the array
stereo_note = np.column_stack((note, note))
# Convert to 16-bit data
audio_array = (stereo_note * 32767).astype(np.int16)
# Create a Sound object
try:
    alert_sound = pygame.mixer.Sound(buffer=audio_array)
    print("Successfully created alert sound")
except:
    print("Failed to create alert sound, will use text alerts only")
    alert_sound = None

# Interactive controls
show_landmarks = True
show_face_box = True
alert_enabled = True

def play_alert_sound():
    """Play alert sound without blocking the main thread"""
    global last_alert_time
    if time.time() - last_alert_time > ALERT_COOLDOWN:
        if alert_sound:
            alert_sound.play()
        print("ALERT: Drowsiness detected!")
        last_alert_time = time.time()

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio"""
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate mouth aspect ratio for yawn detection"""
    # Compute vertical distances
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    # Compute horizontal distance
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    # Compute mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar * 100  # Scale up for better threshold comparison

def draw_status_panel(frame, status, blinks, yawns, ear, mar):
    """Draw a status panel with all information"""
    # Create a semi-transparent overlay for the status panel
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw status text with background
    cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Blinks: {blinks}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawns: {yawns}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw controls help
    cv2.putText(frame, "Press 'l': Toggle landmarks", (w-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'b': Toggle face box", (w-300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'a': Toggle alerts", (w-300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'r': Reset counters", (w-300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'ESC': Exit", (w-300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    global last_alert_time, status, color, blink_count, yawn_count
    global show_landmarks, show_face_box, alert_enabled
    global consecutive_closed, consecutive_yawn, eye_closed, mouth_open
    global sleep_frames, drowsy_frames, active_frames  # Added active_frames to globals
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Create a copy for visualization
        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 0)
        
        if len(faces) == 0:
            status = "No Face Detected"
            color = (200, 200, 200)  # Gray
            draw_status_panel(vis_frame, status, blink_count, yawn_count, 0, 0)
        
        for face in faces:
            # Get face bounding box
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            if show_face_box:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Extract eye coordinates
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]
            
            # Calculate eye aspect ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate mouth aspect ratio
            mar = mouth_aspect_ratio(mouth)
            
            # Draw landmarks if enabled
            if show_landmarks:
                for (x, y) in landmarks:
                    cv2.circle(vis_frame, (x, y), 1, (0, 255, 255), -1)
                
                # Highlight eyes and mouth with different colors
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                
                # Use different colors for eyes and mouth
                cv2.drawContours(vis_frame, [left_eye_hull], -1, (255, 0, 0), 1)  # Blue for left eye
                cv2.drawContours(vis_frame, [right_eye_hull], -1, (255, 0, 0), 1)  # Blue for right eye
                cv2.drawContours(vis_frame, [mouth_hull], -1, (0, 0, 255), 1)  # Red for mouth
            
            # Blink detection
            if ear < EYE_AR_THRESHOLD:
                consecutive_closed += 1
                if consecutive_closed >= EYE_AR_CONSEC_FRAMES and not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                consecutive_closed = 0
                eye_closed = False
            
            # Yawn detection
            if mar > YAWN_THRESHOLD:
                consecutive_yawn += 1
            if consecutive_yawn >= 3 and not mouth_open:
                yawn_count += 1
                mouth_open = True
            elif mar <= YAWN_THRESHOLD and mouth_open:
                mouth_open = False
                consecutive_yawn = 0
            else:
                consecutive_yawn = 0
            
            # Determine drowsiness state
            if ear < EYE_AR_THRESHOLD * 0.8:  # More strict threshold for sleep detection
                sleep_frames += 1
                drowsy_frames = 0
                active_frames = 0
                
                if sleep_frames > SLEEP_THRESHOLD:
                    status = "SLEEPING!"
                    color = (0, 0, 255)  # Red
                    if alert_enabled:
                        play_alert_sound()
            
            elif ear < EYE_AR_THRESHOLD:
                sleep_frames = 0
                drowsy_frames += 1
                active_frames = 0
                
                if drowsy_frames > DROWSY_THRESHOLD:
                    status = "Drowsy!"
                    color = (0, 140, 255)  # Orange
                    if alert_enabled:
                        play_alert_sound()
            
            else:
                sleep_frames = 0
                drowsy_frames = 0
                active_frames += 1
                
                if active_frames > 3:
                    status = "Alert"
                    color = (0, 255, 0)  # Green
            
            # Draw eye and mouth metrics with colored backgrounds
            # Create background for metrics
            cv2.rectangle(vis_frame, (x1, y1-30), (x1+120, y1-5), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (x1, y2+5), (x1+120, y2+30), (0, 0, 0), -1)
            
            cv2.putText(vis_frame, f"EAR: {ear:.2f}", (x1+5, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(vis_frame, f"MAR: {mar:.2f}", (x1+5, y2+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw status panel
            draw_status_panel(vis_frame, status, blink_count, yawn_count, ear, mar)
        
        # Add a title bar
        h, w = vis_frame.shape[:2]
        cv2.rectangle(vis_frame, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(vis_frame, "Driver Drowsiness Detection System", (w//2-200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Drowsiness Detection", vis_frame)
        
        # Handle key presses for interactive controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks display: {'On' if show_landmarks else 'Off'}")
        elif key == ord('b'):
            show_face_box = not show_face_box
            print(f"Face box display: {'On' if show_face_box else 'Off'}")
        elif key == ord('a'):
            alert_enabled = not alert_enabled
            print(f"Alerts: {'Enabled' if alert_enabled else 'Disabled'}")
        elif key == ord('r'):
            blink_count = 0
            yawn_count = 0
            print("Counters reset")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()