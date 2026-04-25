import pandas as pd
import json
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Database Connection
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"

def train_localization_model():
    print("🔌 Connecting to database...")
    engine = create_engine(DB_URL)
    
    # We only pull scans that have a Ground Truth location_id
    query = "SELECT * FROM raw_scans WHERE location_id IS NOT NULL;" 
    
    try:
        df_raw = pd.read_sql(query, engine)
        if len(df_raw) < 10:
            print(f"⚠️ Only {len(df_raw)} labeled scans found. Need more data to train effectively.")
            return
        print(f"✅ Successfully pulled {len(df_raw)} labeled scans.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    print("⚙️ Flattening RF signatures into Feature Matrix...")
    flattened_data = []

    for _, row in df_raw.iterrows():
        # Target Label is now the actual Room/Location name from your dropdown
        scan_features = {'target_location': row['location_id']}
        
        # --- Extract Wi-Fi Signals ---
        try:
            wifi_payload = row['wifi_data']
            if isinstance(wifi_payload, str):
                wifi_payload = json.loads(wifi_payload)
            
            # The Android app sends this as a JSON string inside the 'fingerprint' key
            if isinstance(wifi_payload, dict) and 'fingerprint' in wifi_payload:
                inner_data = json.loads(wifi_payload['fingerprint'])
                for wifi in inner_data.get('wifiInfo', []):
                    bssid = wifi.get('bssid', 'unknown')
                    rssi = wifi.get('rssi', -100)
                    scan_features[f"WIFI_{bssid}"] = rssi
        except Exception:
            pass
            
        # --- Extract Cellular Signals ---
        try:
            cell_payload = row['cell_data']
            if isinstance(cell_payload, str):
                cell_payload = json.loads(cell_payload)
                
            if isinstance(cell_payload, dict) and 'fingerprint' in cell_payload:
                inner_data = json.loads(cell_payload['fingerprint'])
                for cell in inner_data.get('cellInfo', []):
                    cid = cell.get('cid', 'unknown')
                    rsrp = cell.get('rsrp', -100)
                    scan_features[f"CELL_{cid}"] = rsrp
        except Exception:
            pass
            
        flattened_data.append(scan_features)

    # 2. Prepare Data for Scikit-Learn
    df_ml = pd.DataFrame(flattened_data).fillna(-100)
    
    X = df_ml.drop('target_location', axis=1) # Features (Signals)
    y = df_ml['target_location']               # Target (Room Name)
    
    # Save the feature column names (very important for prediction later!)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'model_features.pkl')

    # 3. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"🧠 Training Random Forest on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate Performance
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))

    # 5. Save the Brain
    joblib.dump(model, 'localization_model.pkl')
    print("💾 Model saved as 'localization_model.pkl'")

if __name__ == '__main__':
    train_localization_model()