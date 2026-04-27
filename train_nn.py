import pandas as pd
import sqlite3
import json
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load data from your local development DB
conn = sqlite3.connect('msl_dev.db')
query = "SELECT cell_data, location_id FROM raw_scans"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Pre-process Fingerprints (Aligning with your existing feature set)
# Assuming 'model_features.pkl' contains your BSSIDs and CIDs
model_features = joblib.load('model_features.pkl')

def extract_features(row):
    try:
        data = json.loads(row['cell_data'])
        fingerprint = json.loads(data['fingerprint'])
        signals = {}
        for wifi in fingerprint.get('wifiInfo', []):
            signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
        for cell in fingerprint.get('cellInfo', []):
            signals[f"CELL_{cell['cid']}"] = cell['rsrp']
        
        # Apply Differential RSSI
        if signals:
            max_rssi = max(signals.values())
            return {k: (v - max_rssi) for k, v in signals.items()}
    except:
        return {}
    return {}

processed_data = df.apply(extract_features, axis=1)
X = pd.DataFrame(processed_data.toList(), columns=model_features).fillna(-100)
y = df['location_id']

# 3. Scaling (CRITICAL for Neural Networks)
# NNs require inputs to be normalized to a standard range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train the Small Neural Network (MLP)
# (64, 32) hidden layers provide a balance of depth and speed
snn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

snn.fit(X_scaled, y)

# 5. Export the artifacts
joblib.dump(snn, 'nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ SNN and Scaler generated successfully!")