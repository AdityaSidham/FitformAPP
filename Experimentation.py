import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Create a Streamlit web app
st.title("Fit Form App")
start_detection = st.button("Start Detection")
stop_detection = st.button("Stop Detection")
st.image("Fitform.png")
cap = None
# Initialize video capture
if start_detection:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    stframe4 = st.empty()
    stframe5 = st.empty()

# Initialize counters and stages
left_counter = 0
right_counter = 0
counter = 0
squat_counter = 0
left_stage = None
right_stage = None
shoulder_stage = None
squat_stage = None

# Setup MediaPipe instances
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while start_detection:
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            results = holistic.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                shoulder_angle = calculate_angle(hip, right_shoulder, right_elbow)
                squat_angle = calculate_angle(hip, knee, ankle)

                # Curl counter logic
                if left_angle > 160:
                    left_stage = "down"
                if left_angle < 30 and left_stage == 'down':
                    left_stage = "up"
                    left_counter += 1

                if right_angle > 160:
                    right_stage = "down"
                if right_angle < 30 and right_stage == 'down':
                    right_stage = "up"
                    right_counter += 1

                if shoulder_angle <= 109:
                    shoulder_stage = "down"
                if shoulder_angle > 169 and shoulder_stage == 'down':
                    shoulder_stage = "up"
                    counter += 1

                if squat_angle > 150:
                    squat_stage = "up"
                if squat_angle < 130 and squat_stage == 'up':
                    squat_stage = "down"
                    squat_counter += 1

            except:
                pass

            # Display video feed and results
            stframe.image(image, channels="BGR", use_column_width=True)
            stframe2.write(f"Left Bicep Counter: {left_counter}")
            stframe3.write(f"Right Bicep Counter: {right_counter}")
            stframe4.write(f"Shoulder Counter: {counter}")
            stframe5.write(f"Squat Counter: {squat_counter}")

            if stop_detection:
                break

if cap is not None:
    cap.release() 