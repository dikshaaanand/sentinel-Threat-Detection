import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.preprocessing import LabelEncoder

# Load saved models and features
print("Loading models...")
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Load dataset
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

# Encode text columns
le_protocol = LabelEncoder()
le_service  = LabelEncoder()
le_flag     = LabelEncoder()
df['protocol_type'] = le_protocol.fit_transform(df['protocol_type'])
df['service']       = le_service.fit_transform(df['service'])
df['flag']          = le_flag.fit_transform(df['flag'])

X = df[features]

# Take a small sample
print("Running SHAP explainer on 100 samples...")
sample = X.sample(100, random_state=42)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(sample)

# Debug shape
print("Checking SHAP output shape...")
shap_array = np.array(shap_values)
print("SHAP array shape:", shap_array.shape)

# Handle all possible shapes
try:
    if shap_array.ndim == 3:
        # Shape is (n_classes, n_samples, n_features)
        # Average across classes for first sample
        shap_for_sample = np.mean(np.abs(shap_array[:, 0, :]), axis=0)

    elif shap_array.ndim == 2:
        # Shape is (n_samples, n_features)
        shap_for_sample = shap_array[0]

    else:
        # Fallback — use feature importances from model directly
        print("Using feature importances as fallback...")
        shap_for_sample = rf_model.feature_importances_

    # Make sure lengths match
    if len(shap_for_sample) != len(features):
        print("Length mismatch, using feature importances instead...")
        shap_for_sample = rf_model.feature_importances_

    # Show prediction and explanation
    prediction = rf_model.predict(sample.iloc[[0]])[0]
    print("\n--- XAI Explanation for Sample Threat ---")
    print(f"Predicted threat: {prediction}")
    print("\nTop features driving this prediction:")

    importance = pd.DataFrame({
        'feature': features,
        'shap_value': shap_for_sample
    }).sort_values('shap_value', ascending=False)

    print(importance.to_string(index=False))

except Exception as e:
    print(f"SHAP failed with: {e}")
    print("Falling back to built-in feature importances...")

    prediction = rf_model.predict(sample.iloc[[0]])[0]
    print(f"\nPredicted threat: {prediction}")
    print("\nTop features driving this prediction:")

    importance = pd.DataFrame({
        'feature': features,
        'shap_value': rf_model.feature_importances_
    }).sort_values('shap_value', ascending=False)

    print(importance.to_string(index=False))

# Save explainer
print("\nSaving SHAP explainer...")
with open('explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

print("\n✅ Done! Explainer saved successfully.")