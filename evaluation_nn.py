import pandas as pd
import json
import joblib
import time
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 1. DATABASE CONNECTION
# Connection to the Render Dashboard
DATABASE_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39" 
engine = create_engine(DATABASE_URL)

# 2. FETCH LABELED DATA FROM RAW_SCANS
print("🌐 Connecting to Render PostgreSQL for Evaluation...")
query = "SELECT cell_data, location_id FROM raw_scans WHERE location_id IS NOT NULL"
df = pd.read_sql_query(query, engine)

print(f"📊 Retrieved {len(df)} samples for evaluation.")

if len(df) < 5:
    print("❌ Not enough data samples to perform a valid statistical evaluation.")
else:
    # 3. LOAD ARTIFACTS
    model = joblib.load('nn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')

    # 4. PRE-PROCESSING (Matches train_nn.py logic)
    def extract_nn_features(row):
        try:
            # Parse outer cell_data
            outer_data = row['cell_data']
            if isinstance(outer_data, str):
                outer_data = json.loads(outer_data)
            
            # Parse inner fingerprint string
            fingerprint_str = outer_data.get('fingerprint', '{}')
            signals_data = json.loads(fingerprint_str)
            
            signals = {}
            for wifi in signals_data.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in signals_data.get('cellInfo', []):
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']
            
            if signals:
                # Differential RSSI Normalization
                max_rssi = max(signals.values())
                return {k: (v - max_rssi) for k, v in signals.items()}
        except:
            return {}
        return {}

    processed_data = df.apply(extract_nn_features, axis=1)
    X = pd.DataFrame(processed_data.tolist(), columns=model_features).fillna(-100)
    y = df['location_id']

    # 5. METRICS CALCULATION
    # 80/20 Split for Accuracy check
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    # A. Accuracy Calculation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # B. Variance (σ²) using 5-Fold Cross-Validation
    X_all_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_scaled, y, cv=5)
    variance = np.var(cv_scores)

    # C. Inference Latency (Mean time for a single prediction)
    start_time = time.time()
    for _ in range(100):
        model.predict(X_test_scaled[0:1])
    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / 100) * 1000

    # 6. PRINT OFFICIAL REPORT
    print("\n" + "="*45)
    print("      INDOR LOCALIZATION: NN PERFORMANCE")
    print("="*45)
    print(f"🎯 Accuracy:           {accuracy * 100:.2f}%")
    print(f"📉 Model Variance (σ²): {variance:.6f}")
    print(f"⚡ Avg Latency:        {avg_latency_ms:.4f} ms")
    print("="*45)
    print("\nDetailed Room Breakdown (Precision/Recall):")
    print(classification_report(y_test, y_pred))