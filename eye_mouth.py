import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd

# Load shape predictor for facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the threshold values for EAR and MAR
EAR_THRESHOLD = 0.25  # Adjust this threshold value as needed
MAR_THRESHOLD_YAWN = 30  # Threshold for detecting yawn
MAR_THRESHOLD_NO_YAWN = 30  # Threshold for no yawn

# Initialize lists to store EAR, MAR, and output values
ear_values = []
mar_values = []
eyes_output = []
mouth_output = []

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to determine if eye is closed based on EAR
def is_eye_closed(ear):
    if ear < EAR_THRESHOLD:
        return True
    else:
        return False

# Function to calculate MAR
def mouth_aspect_ratio(mouth): 
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    MAR = (A + B + C) / 3.0
    return MAR

# Function to determine mouth state based on MAR
def is_mouth_open(mar):
    if mar > MAR_THRESHOLD_YAWN:
        return "Yawn"
    elif mar < MAR_THRESHOLD_NO_YAWN:
        return "No Yawn"
    else:
        return "Unknown"

# Initialize video capture from mobile camera stream
mobile_camera_url = 'http://192.168.6.47:8080/video'
cap = cv2.VideoCapture(mobile_camera_url)

while True:
    # Read frame from video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect facial landmarks
        landmarks = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract eye landmarks
        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]

        # Calculate EAR for left and right eyes
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)

        # Determine eye state (open/closed)
        left_eye_closed = is_eye_closed(left_ear)
        right_eye_closed = is_eye_closed(right_ear)

        # Add average EAR value to list
        ear_values.append((left_ear + right_ear) / 2)  # Average EAR for both eyes

        # Determine overall eye state
        if left_eye_closed and right_eye_closed:
            eyes_output.append("Closed")
        else:
            eyes_output.append("Open")

        # Display eye state on the frame
        left_eye_state = "Closed" if left_eye_closed else "Open"
        right_eye_state = "Closed" if right_eye_closed else "Open"
        cv2.putText(frame, f"Left Eye: {left_eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Eye: {right_eye_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Extract mouth landmarks
        mouth_landmarks = landmarks[48:68]

        # Calculate MAR
        mar = mouth_aspect_ratio(mouth_landmarks)

        # Determine mouth state (yawn/no yawn)
        mouth_state = is_mouth_open(mar)

        # Add MAR value to list
        mar_values.append(mar)

        # Add mouth state to output list
        mouth_output.append(mouth_state)

        # Display MAR on the frame
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Create a DataFrame to store EAR, MAR, and eye/mouth output values
data = {"EAR": ear_values, "MAR": mar_values, "Eyes Output": eyes_output, "Mouth Output": mouth_output}
df = pd.DataFrame(data)

# Save DataFrame to Excel file
df.to_excel("output_data.xlsx", index=False)
