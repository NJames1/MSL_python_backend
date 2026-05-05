import ast
import json
import pandas as pd
from db import SessionLocal
import models

def export_real_scans():
    # 1. Open Database Connection
    db = SessionLocal()
    
    try:
        print("📡 Connecting to database and fetching scans...")
        
        # 2. Query all scans that have an actual labeled location
        # We ignore "Unknown" because we only want Anchor Point data for training
        scans = db.query(models.RawScan).filter(models.RawScan.location_id != "Unknown").all()
        
        if not scans:
            print("❌ No labeled scans found. Check if the database has data.")
            return

        dataset = []
        
        # 3. Extract and Flatten the JSON data from cell_data
        for scan in scans:
            row = {'location_id': scan.location_id}
            
            # The data is actually hiding inside the cell_data column!
            raw_cell_data = scan.cell_data
            
            if raw_cell_data:
                try:
                    # 1. Make sure cell_data is a dictionary
                    if isinstance(raw_cell_data, str):
                        cell_data_dict = json.loads(raw_cell_data)
                    else:
                        cell_data_dict = raw_cell_data
                        
                    # 2. Extract the "fingerprint" string (as seen in your screenshot)
                    fingerprint_str = cell_data_dict.get('fingerprint')
                    
                    if fingerprint_str:
                        # 3. The fingerprint itself is a stringified JSON, so we parse it again
                        fingerprint_dict = json.loads(fingerprint_str)
                        
                        # 4. FINALLY, extract the Wi-Fi list!
                        wifi_list = fingerprint_dict.get('wifiInfo', [])
                        
                        # 5. Map the data to our CSV columns
                        for wifi in wifi_list:
                            if isinstance(wifi, dict) and 'bssid' in wifi:
                                column_name = f"WIFI_{wifi['bssid']}"
                                row[column_name] = wifi.get('rssi', -100)
                                
                except Exception as e:
                    print(f"⚠️ Skipping scan ID {scan.id}: Extraction error ({e})")
                    continue
            
            # Only append rows that actually found Wi-Fi data
            # (Checks if the row has more than just the 'location_id' key)
            if len(row) > 1:
                dataset.append(row)
            
            dataset.append(row)
        # 4. Convert to a Pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # 5. Handle missing values
        # If a phone didn't see a specific router during a scan, we set it to -100 dBm
        df = df.fillna(-100)
        
        # 6. Save to CSV
        output_file = 'real_mamlaka_aw_data.csv'
        df.to_csv(output_file, index=False)
        
        print(f"✅ Success! Exported {len(df)} scans to {output_file}")
        print(f"📊 Total Unique Wi-Fi Features extracted: {len(df.columns) - 1}")

    except Exception as e:
        print(f"💥 Database Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    export_real_scans()