import pandas as pd
import json
import joblib
import time
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

# 1. Database Connection
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"

def train_final_production_model():
    print("🔌 Pulling data from Render PostgreSQL...")
    engine = create_engine(DB_URL)
    query = "SELECT * FROM raw_scans WHERE location_id IS NOT NULL;" 
    df_raw = pd.read_sql(query, engine)
    
    # Filter for high-confidence classrooms and entrance
    exclude = ['Corridor', 'Electronics Lab', 'Machines Lab', 'Controls Lab', 'Engineering Gate']
    df_filtered = df_raw[~df_raw['location_id'].str.contains('|'.join(exclude), case=False, na=False)]

    print(f"⚙️ Applying Differential RSSI Algorithm...")
    processed_scans = []

    for _, row in df_filtered.iterrows():
        try:
            raw_payload = row['cell_data']
            if isinstance(raw_payload, str): raw_payload = json.loads(raw_payload)
            inner = json.loads(raw_payload['fingerprint'])
            
            signals = {}
            for wifi in inner.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in inner.get('cellInfo', []):
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']
            
            if not signals: continue

            # --- THE "SECRET SAUCE": Differential Normalization ---
            max_rssi = max(signals.values())
            diff_features = {k: (v - max_rssi) for k, v in signals.items()}
            diff_features['target_location'] = row['location_id']
            processed_scans.append(diff_features)
        except: continue

    df_ml = pd.DataFrame(processed_scans).fillna(-100)
    X = df_ml.drop('target_location', axis=1)
    y = df_ml['target_location']

    # 2. The Model: Random Forest
    model_name = "Random Forest Classifier (Ensemble)"
    rf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        min_samples_leaf=3, 
        random_state=42, 
        class_weight='balanced'
    )

    # 3. Scientific Validation (Cross-Validation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=skf)

    # 4. Latency Benchmarking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    
    start_time = time.perf_counter()
    _ = rf.predict(X_test)
    end_time = time.perf_counter()
    
    # Calculate average time per scan in milliseconds
    latency_ms = ((end_time - start_time) / len(X_test)) * 1000

    # 5. Output Final Results
    print("\n" + "="*45)
    print("🎯 FINAL PROJECT PERFORMANCE BENCHMARK")
    print("="*45)
    print(f"Classification Model: {model_name}")
    print(f"Mean Accuracy (CV):   {cv_scores.mean():.2%}")
    print(f"Variance (95% CI):    (+/- {cv_scores.std() * 2:.2%})")
    print(f"Inference Latency:    {latency_ms:.4f} ms per scan")
    print("="*45)

    # Save the "Brain" and the Feature Map
    joblib.dump(X.columns.tolist(), 'model_features.pkl')
    joblib.dump(rf, 'localization_model.pkl')
    print(f"\n💾 Production model successfully deployed locally.")

if __name__ == '__main__':
    train_final_production_model()