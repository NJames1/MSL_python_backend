from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/msl_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Initialize database tables. Imports models so metadata is registered."""
    # import models to ensure they are registered on the Base metadata
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session for dependency injection in FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
