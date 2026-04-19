from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import cv2
import base64

app = Flask(__name__)
CORS(app)

with open('workout_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

import mediapipe as mp
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(r * 180 / np.pi)
    return 360 - angle if angle > 180 else angle

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

REP_ANGLES = {
    'pushup':     ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 90),
    'pullup':     ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 60),
    'squat':      ('LEFT_HIP',     'LEFT_KNEE',  'LEFT_ANKLE',  170, 90),
    'bicep_curl': ('LEFT_SHOULDER','LEFT_ELBOW', 'LEFT_WRIST',  160, 30),
    'deadlift':   ('LEFT_SHOULDER','LEFT_HIP',   'LEFT_KNEE',   160, 70),
}

rep_state = {'counter': 0, 'stage': None, 'last_ex': None}

@app.route('/')
def home():
    return "Workout Pose API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.json
        img_data = base64.b64decode(data['image'].split(',')[1])
        np_arr   = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = pose.process(rgb)

        if not results.pose_landmarks:
            return jsonify({
                'exercise': 'No pose detected',
                'confidence': 0,
                'reps': rep_state['counter'],
                'stage': rep_state['stage']
            })

        lm       = results.pose_landmarks.landmark
        features = get_features(lm)
        exercise = model.predict([features])[0]
        conf     = float(max(model.predict_proba([features])[0]))

        if exercise != rep_state['last_ex']:
            rep_state['counter'] = 0
            rep_state['stage']   = None
            rep_state['last_ex'] = exercise

        if exercise in REP_ANGLES:
            a, b, c, up, dn = REP_ANGLES[exercise]
            def gp(n):
                pt = lm[mp_pose.PoseLandmark[n].value]
                return [pt.x, pt.y]
            ang = calculate_angle(gp(a), gp(b), gp(c))
            if ang > up:
                rep_state['stage'] = 'up'
            if ang < dn and rep_state['stage'] == 'up':
                rep_state['stage']   = 'down'
                rep_state['counter'] += 1

        return jsonify({
            'exercise':   exercise,
            'confidence': round(conf, 2),
            'reps':       rep_state['counter'],
            'stage':      rep_state['stage']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    rep_state['counter'] = 0
    rep_state['stage']   = None
    rep_state['last_ex'] = None
    return jsonify({'message': 'Reset successful'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)