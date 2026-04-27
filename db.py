from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# --- DATABASE CONFIGURATION ---
# Replace the placeholder below with your "External Database URL" from the Render dashboard.
# Example: "postgresql://user:password@hostname.render.com/msl_live_demo"
LIVE_DB_URL = 'postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39'
'

# Logic to prioritize the environment variable (for Render deployment) 
# while falling back to your new live URL for local testing.
DATABASE_URL = os.getenv("DATABASE_URL", LIVE_DB_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    """
    Initialize database tables in the new instance. 
    Metadata registration ensures the 'live_test_results' table is created.
    """
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session for dependency injection in FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()