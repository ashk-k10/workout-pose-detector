import cv2, pickle, numpy as np
import mediapipe as mp

with open('workout_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(r*180/np.pi)
    return 360-angle if angle>180 else angle

def get_features(lm):
    def p(n):
        pt = lm[mp_pose.PoseLandmark[n].value]
        return [pt.x, pt.y]
    return [
        calculate_angle(p('LEFT_SHOULDER'), p('LEFT_ELBOW'),    p('LEFT_WRIST')),
        calculate_angle(p('RIGHT_SHOULDER'),p('RIGHT_ELBOW'),   p('RIGHT_WRIST')),
        calculate_angle(p('LEFT_ELBOW'),    p('LEFT_SHOULDER'), p('LEFT_HIP')),
        calculate_angle(p('RIGHT_ELBOW'),   p('RIGHT_SHOULDER'),p('RIGHT_HIP')),
        calculate_angle(p('LEFT_SHOULDER'), p('LEFT_HIP'),      p('LEFT_KNEE')),
        calculate_angle(p('RIGHT_SHOULDER'),p('RIGHT_HIP'),     p('RIGHT_KNEE')),
        calculate_angle(p('LEFT_HIP'),      p('LEFT_KNEE'),     p('LEFT_ANKLE')),
        calculate_angle(p('RIGHT_HIP'),     p('RIGHT_KNEE'),    p('RIGHT_ANKLE')),
        calculate_angle(p('LEFT_KNEE'),     p('LEFT_ANKLE'),    p('LEFT_HIP')),
        calculate_angle(p('RIGHT_KNEE'),    p('RIGHT_ANKLE'),   p('RIGHT_HIP')),
    ]

REP = {
    'pushup':     ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 90),
    'pullup':     ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 60),
    'squat':      ('LEFT_HIP',     'LEFT_KNEE',  'LEFT_ANKLE',  170, 90),
    'bicep_curl': ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 30),
    'deadlift':   ('LEFT_SHOULDER','LEFT_HIP',   'LEFT_KNEE',   160, 70),
}

counter, stage, last_ex = 0, None, None
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        exercise, conf = "No pose", 0.0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                feat     = get_features(lm)
                exercise = model.predict([feat])[0]
                conf     = max(model.predict_proba([feat])[0])

                if exercise != last_ex:
                    counter, stage, last_ex = 0, None, exercise

                if exercise in REP:
                    a, b, c, up, dn = REP[exercise]
                    def gp(n):
                        pt = lm[mp_pose.PoseLandmark[n].value]
                        return [pt.x, pt.y]
                    ang = calculate_angle(gp(a), gp(b), gp(c))
                    if ang > up: stage = "up"
                    if ang < dn and stage == "up":
                        stage = "down"; counter += 1
            except:
                pass

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(255,255,255), thickness=2))

        cv2.rectangle(frame, (0,0), (frame.shape[1], 85), (0,0,0), -1)
        cv2.putText(frame, f"Exercise: {exercise} ({conf:.0%})",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,255), 2)
        cv2.putText(frame, f"Reps: {counter}  Stage: {stage}",
                    (10,62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

        cv2.imshow('Workout Detector - press Q to quit', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): counter, stage = 0, None

cap.release()
cv2.destroyAllWindows()