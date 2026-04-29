import sqlalchemy as sa
from sqlalchemy import create_engine, text
import json

# 1. USE YOUR EXTERNAL DATABASE URL
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"
engine = create_engine(DB_URL)

def backfill_serving_cells():
    with engine.connect() as conn:
        print("🔍 Fetching rows with missing serving_cell...")
        # Get ID and the JSON data for rows that are currently None
        result = conn.execute(text("SELECT id, cell_data FROM raw_scans WHERE serving_cell IS NULL;"))
        rows = result.fetchall()
        
        if not rows:
            print("✅ No empty cells found. Your database is already up to date!")
            return

        print(f"⚙️ Processing {len(rows)} records...")
        updated_count = 0

        for row_id, cell_data in rows:
            try:
                serving_cid = None
                
                # Step A: Handle potential string vs dict format
                if isinstance(cell_data, str):
                    cell_data = json.loads(cell_data)
                
                # Step B: Extract the nested fingerprint string
                # Based on your api.py logic: cell_data = {"fingerprint": "..."}
                fingerprint_str = cell_data.get('fingerprint', '{}')
                signals_data = json.loads(fingerprint_str)
                
                # Step C: Find the serving CID
                cell_info = signals_data.get('cellInfo', [])
                if cell_info:
                    # Look for registered tower, fallback to the first one
                    serving_cid = next((c.get('cid') for c in cell_info if c.get('isRegistered')), cell_info[0].get('cid'))

                # Step D: Update the database
                if serving_cid:
                    conn.execute(
                        text("UPDATE raw_scans SET serving_cell = :cid WHERE id = :id"),
                        {"cid": serving_cid, "id": row_id}
                    )
                    updated_count += 1
            
            except Exception as e:
                print(f"⚠️ Error processing row {row_id}: {e}")
                continue
        
        conn.commit()
        print(f"🎉 Success! Updated {updated_count} rows with serving Cell IDs.")

if __name__ == "__main__":
    backfill_serving_cells()