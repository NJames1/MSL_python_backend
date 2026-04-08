import pandas as pd
import json
from sqlalchemy import create_engine

# 1. Database Connection (Ensure it starts with postgresql://)
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"

def extract_and_flatten_data():
    print("🔌 Connecting to database...")
    engine = create_engine(DB_URL)
    
    # Pulling from the table we verified has your data
    query = "SELECT * FROM raw_scans;" 
    
    try:
        df_raw = pd.read_sql(query, engine)
        print(f"✅ Successfully pulled {len(df_raw)} raw scans.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

    print("⚙️ Flattening nested JSON into ML Feature Matrix...")
    flattened_data = []

    for index, row in df_raw.iterrows():
        # Since there is no 'location_id', we will combine the GPS coordinates to act as our 'Target Label'
        label = f"Lat:{row['gps_lat']}_Lon:{row['gps_lon']}"
        
        scan_features = {'target_location': label}
        
        # --- Extract Wi-Fi Signals ---
        try:
            wifi_payload = row['wifi_data']
            # Convert string to dictionary if necessary
            if isinstance(wifi_payload, str):
                wifi_payload = json.loads(wifi_payload)
                
            if isinstance(wifi_payload, list):
                for wifi in wifi_payload:
                    bssid = wifi.get('bssid', 'unknown')
                    rssi = wifi.get('rssi', -100)
                    scan_features[f"WIFI_{bssid}"] = rssi
        except Exception:
            pass # Gracefully skip any corrupted rows
            
        # --- Extract Cellular Signals ---
        try:
            cell_payload = row['cell_data']
            # Convert string to dictionary if necessary
            if isinstance(cell_payload, str):
                cell_payload = json.loads(cell_payload)
                
            if isinstance(cell_payload, list):
                for cell in cell_payload:
                    cid = cell.get('cid', 'unknown')
                    rsrp = cell.get('rsrp', -100)
                    scan_features[f"CELL_{cid}"] = rsrp
        except Exception:
            pass # Gracefully skip any corrupted rows
            
        flattened_data.append(scan_features)

    # Convert to a Machine Learning ready DataFrame
    df_ml = pd.DataFrame(flattened_data)

    # ML models crash on missing data. Fill unseen routers with -100 dBm (dead signal)
    df_ml.fillna(-100, inplace=True)

    print("\n🚀 Feature Matrix Extraction Complete!")
    print("-" * 50)
    print(f"Total Samples (Rows): {df_ml.shape[0]}")
    print(f"Total Unique RF Signals (Columns): {df_ml.shape[1] - 1}") 
    print("-" * 50)
    
    return df_ml

# Execute the extraction
ml_dataframe = extract_and_flatten_data()

# Preview the beautiful, flattened matrix!
if ml_dataframe is not None:
    print(ml_dataframe.head())