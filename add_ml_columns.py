import sqlalchemy as sa
from sqlalchemy import create_engine, text

# 🛑 IMPORTANT: Paste your EXTERNAL Render Database URL here
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"
engine = create_engine(DB_URL)

def add_prediction_columns():
    commands = [
        "ALTER TABLE raw_scans ADD COLUMN IF NOT EXISTS rf_prediction VARCHAR DEFAULT 'Unknown';",
        "ALTER TABLE raw_scans ADD COLUMN IF NOT EXISTS nn_prediction VARCHAR DEFAULT 'Unknown';"
    ]
    
    with engine.connect() as conn:
        print("🚀 Updating database schema...")
        for cmd in commands:
            try:
                conn.execute(text(cmd))
                conn.commit()
                print(f"✅ Success: Added column")
            except Exception as e:
                print(f"⚠️ Note: {e}")
                
        print("\n🎉 Database updated! The API can now save predictions.")

if __name__ == "__main__":
    add_prediction_columns()