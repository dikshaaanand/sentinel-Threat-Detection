from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import pickle
import random
import threading
import time
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentinel-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
print("Loading models...")
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('iso_model.pkl', 'rb') as f:
    iso_model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

print("All models loaded!")

# Load and prepare dataset for sampling
df = pd.read_csv('KDDTrain+.txt', header=None)
df.columns = [
    'duration','protocol_type','service','flag','src_bytes',
    'dst_bytes','land','wrong_fragments','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell',
    'su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

le_protocol = LabelEncoder()
le_service  = LabelEncoder()
le_flag     = LabelEncoder()
df['protocol_type'] = le_protocol.fit_transform(df['protocol_type'])
df['service']       = le_service.fit_transform(df['service'])
df['flag']          = le_flag.fit_transform(df['flag'])

X = df[features]

def get_shap_explanation(sample_row):
    try:
        shap_values = explainer.shap_values(sample_row)
        shap_array  = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_for_sample = np.mean(np.abs(shap_array[:, 0, :]), axis=0)
        elif shap_array.ndim == 2:
            shap_for_sample = shap_array[0]
        else:
            shap_for_sample = rf_model.feature_importances_

        if len(shap_for_sample) != len(features):
            shap_for_sample = rf_model.feature_importances_

    except:
        shap_for_sample = rf_model.feature_importances_

    explanation = []
    for feat, val in zip(features, shap_for_sample):
        explanation.append({
            'feature': feat,
            'value': round(float(val), 4),
            'direction': 'pos' if val > 0 else 'neg'
        })

    explanation.sort(key=lambda x: abs(x['value']), reverse=True)
    return explanation[:6]

def generate_threat():
    sample = X.sample(1, random_state=random.randint(0, 9999))
    prediction  = rf_model.predict(sample)[0]
    iso_score   = iso_model.decision_function(sample)[0]
    risk_score  = max(0, min(100, int((1 - iso_score) * 50 + random.randint(20, 50))))
    explanation = get_shap_explanation(sample)

    severities = {
        'DDOS':       'critical',
        'EXPLOIT':    'critical',
        'CREDENTIAL': 'high',
        'PORTSCAN':   'high',
        'NORMAL':     'low'
    }

    ips = [
        '185.220.101.47','103.42.56.11','91.108.4.55',
        '198.51.100.23','203.0.113.99','45.33.32.156',
        '192.168.1.'+str(random.randint(1,255))
    ]

    return {
        'id':          random.randint(1000, 9999),
        'threat_type': prediction,
        'severity':    severities.get(prediction, 'medium'),
        'ip':          random.choice(ips),
        'risk_score':  risk_score,
        'explanation': explanation,
        'timestamp':   time.strftime('%H:%M:%S')
    }

# API routes
@app.route('/')
def index():
    return jsonify({'status': 'Sentinel API running', 'models': 'loaded'})

@app.route('/predict', methods=['GET'])
def predict():
    threat = generate_threat()
    return jsonify(threat)

@app.route('/predict/batch', methods=['GET'])
def predict_batch():
    threats = [generate_threat() for _ in range(5)]
    return jsonify(threats)

# Background thread — emits a new threat every few seconds via WebSocket
def threat_stream():
    while True:
        time.sleep(random.randint(3, 7))
        threat = generate_threat()
        socketio.emit('new_threat', threat)
        print(f"[{threat['timestamp']}] Emitted: {threat['threat_type']} from {threat['ip']}")

@socketio.on('connect')
def on_connect():
    print("Client connected!")
    emit('status', {'message': 'Connected to Sentinel'})

if __name__ == '__main__':
    # Start background threat stream
    t = threading.Thread(target=threat_stream, daemon=True)
    t.start()
    print("\n✅ Sentinel API running at http://localhost:5000")
    print("Open your browser and go to http://localhost:5000/predict")
    socketio.run(app, debug=False, port=5000)