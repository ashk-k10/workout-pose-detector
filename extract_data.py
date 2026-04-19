import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def extract_features(landmarks):
    def lm(name):
        p = landmarks[mp_pose.PoseLandmark[name].value]
        return [p.x, p.y]
    return [
        calculate_angle(lm('LEFT_SHOULDER'),  lm('LEFT_ELBOW'),   lm('LEFT_WRIST')),
        calculate_angle(lm('RIGHT_SHOULDER'), lm('RIGHT_ELBOW'),  lm('RIGHT_WRIST')),
        calculate_angle(lm('LEFT_ELBOW'),     lm('LEFT_SHOULDER'),lm('LEFT_HIP')),
        calculate_angle(lm('RIGHT_ELBOW'),    lm('RIGHT_SHOULDER'),lm('RIGHT_HIP')),
        calculate_angle(lm('LEFT_SHOULDER'),  lm('LEFT_HIP'),     lm('LEFT_KNEE')),
        calculate_angle(lm('RIGHT_SHOULDER'), lm('RIGHT_HIP'),    lm('RIGHT_KNEE')),
        calculate_angle(lm('LEFT_HIP'),       lm('LEFT_KNEE'),    lm('LEFT_ANKLE')),
        calculate_angle(lm('RIGHT_HIP'),      lm('RIGHT_KNEE'),   lm('RIGHT_ANKLE')),
        calculate_angle(lm('LEFT_KNEE'),      lm('LEFT_ANKLE'),   lm('LEFT_HIP')),
        calculate_angle(lm('RIGHT_KNEE'),     lm('RIGHT_ANKLE'),  lm('RIGHT_HIP')),
    ]

os.makedirs('data', exist_ok=True)
csv_file = 'data/training_data.csv'

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['a1','a2','a3','a4','a5',
                     'a6','a7','a8','a9','a10','label'])

# Reads every video in every subfolder of videos/
video_base = 'videos'
for exercise in os.listdir(video_base):
    exercise_path = os.path.join(video_base, exercise)
    if not os.path.isdir(exercise_path):
        continue
    for video_file in os.listdir(exercise_path):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(exercise_path, video_file)
        cap = cv2.VideoCapture(video_path)
        count = 0
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    try:
                        features = extract_features(
                            results.pose_landmarks.landmark)
                        with open(csv_file, 'a', newline='') as f:
                            csv.writer(f).writerow(
                                features + [exercise])
                        count += 1
                    except:
                        pass
        cap.release()
        print(f"{exercise} / {video_file} → {count} frames extracted")

print("\nDone! Check data/training_data.csv")