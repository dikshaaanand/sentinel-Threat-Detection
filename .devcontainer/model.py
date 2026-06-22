import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

print("Loading dataset...")
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

# Simplify labels into threat categories
print("Cleaning data...")
def map_label(label):
    if label == 'normal': return 'NORMAL'
    elif label in ['neptune','smurf','pod','teardrop','land','back','apache2','udpstorm','processtable','mailbomb']: return 'DDOS'
    elif label in ['satan','ipsweep','nmap','portsweep','mscan','saint']: return 'PORTSCAN'
    elif label in ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']: return 'CREDENTIAL'
    else: return 'EXPLOIT'

df['threat'] = df['label'].apply(map_label)

# Encode text columns
le_protocol = LabelEncoder()
le_service  = LabelEncoder()
le_flag     = LabelEncoder()

df['protocol_type'] = le_protocol.fit_transform(df['protocol_type'])
df['service']       = le_service.fit_transform(df['service'])
df['flag']          = le_flag.fit_transform(df['flag'])

# Features and labels
features = ['duration','protocol_type','service','flag','src_bytes',
            'dst_bytes','wrong_fragments','hot','num_failed_logins',
            'logged_in','num_compromised','count','srv_count',
            'serror_rate','rerror_rate','same_srv_rate','diff_srv_rate']

X = df[features]
y = df['threat']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate
print("\n--- Model Results ---")
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Train Isolation Forest for anomaly detection
print("Training Isolation Forest anomaly detector...")
iso_model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
iso_model.fit(X_train)

# Save both models
print("Saving models...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('iso_model.pkl', 'wb') as f:
    pickle.dump(iso_model, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\n✅ Models trained and saved successfully!")