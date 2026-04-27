import pandas as pd
import json
import joblib
from sqlalchemy import create_engine
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. DATABASE CONNECTION
# Use your External Connection String from Render
DATABASE_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39" 
engine = create_engine(DATABASE_URL)

# 2. FETCH DATA FROM RAW_SCANS
# We only want rows where location_id is NOT NULL (our labeled training data)
print("🌐 Connecting to Render PostgreSQL...")
query = "SELECT cell_data, location_id FROM raw_scans WHERE location_id IS NOT NULL"
df = pd.read_sql_query(query, engine)

print(f"📊 Successfully retrieved {len(df)} samples from raw_scans.")

if len(df) == 0:
    print("❌ Still 0 samples. Check if your table name is 'raw_scans' and has data.")
else:
    # 3. PRE-PROCESSING
    model_features = joblib.load('model_features.pkl')

    def extract_nn_features(row):
        try:
            # Step A: Parse the outer 'cell_data' JSON
            outer_data = row['cell_data']
            if isinstance(outer_data, str):
                outer_data = json.loads(outer_data)
            
            # Step B: Parse the inner 'fingerprint' string
            fingerprint_str = outer_data.get('fingerprint', '{}')
            signals_data = json.loads(fingerprint_str)
            
            signals = {}
            for wifi in signals_data.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in signals_data.get('cellInfo', []):
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']
            
            if signals:
                # Hardware-Agnostic Normalization
                max_rssi = max(signals.values())
                return {k: (v - max_rssi) for k, v in signals.items()}
        except Exception as e:
            return {}
        return {}

    processed_data = df.apply(extract_nn_features, axis=1)
    X = pd.DataFrame(processed_data.tolist(), columns=model_features).fillna(-100)
    y = df['location_id']

    # 4. SCALING & TRAINING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    snn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    print("🧠 Training the Neural Network...")
    snn.fit(X_scaled, y)

    # 5. EXPORT
    joblib.dump(snn, 'nn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Success! Neural Network and Scaler are now ready.")