from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True)
    device_hash = Column(String(64), unique=True, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    
    # Establish relationship to scans
    scans = relationship("RawScan", back_populates="device")


class RawScan(Base):
    __tablename__ = "raw_scans"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=True)
    user_name = Column(String)
    
    # This remains as your "Ground Truth" (what you select in the app)
    location_id = Column(String)
    
    # NEW: Storage for model predictions to allow for side-by-side comparison
    rf_prediction = Column(String, nullable=True) 
    nn_prediction = Column(String, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow)
    cell_data = Column(JSON)
    wifi_data = Column(JSON)
    
    # Using Numeric for high precision required for coordinate mapping
    gps_lat = Column(Numeric(precision=10, scale=6), nullable=True)
    gps_lon = Column(Numeric(precision=10, scale=6), nullable=True)

    # Establish relationship back to device
    device = relationship("Device", back_populates="scans")


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    centroid_lat = Column(Numeric(precision=10, scale=6))
    centroid_lon = Column(Numeric(precision=10, scale=6))
    created_at = Column(DateTime, default=datetime.utcnow)


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    features = Column(JSON)
    confidence = Column(Numeric(precision=5, scale=4), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)