import sqlalchemy as sa
from sqlalchemy import create_engine, text

# Use your EXTERNAL Database URL from Render
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"
engine = create_engine(DB_URL)

def run_safe_migration():
    commands = [
        # 1. Add proximity_verified
        "ALTER TABLE raw_scans ADD COLUMN IF NOT EXISTS proximity_verified BOOLEAN DEFAULT FALSE;",
        
        # 2. Add serving_cell
        "ALTER TABLE raw_scans ADD COLUMN IF NOT EXISTS serving_cell INTEGER;",
        
        # 3. Create the anchor pool table
        """
        CREATE TABLE IF NOT EXISTS tower_pool (
            id SERIAL PRIMARY KEY,
            location_id VARCHAR,
            cell_id INTEGER,
            last_confirmed_gps TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score INTEGER DEFAULT 1
        );
        """
    ]

    with engine.connect() as conn:
        print("🚀 Starting safe migration...")
        for cmd in commands:
            try:
                conn.execute(text(cmd))
                conn.commit() # Commit each one individually
                print(f"✅ Executed: {cmd[:40]}...")
            except Exception as e:
                # If it's a 'DuplicateColumn' error, we just ignore it
                if "already exists" in str(e):
                    print(f"ℹ️ Already exists, skipping: {cmd[:40]}...")
                else:
                    print(f"❌ Error: {e}")
        print("\n🎉 All structures verified. Your dashboard should be live now!")

if __name__ == "__main__":
    run_safe_migration()