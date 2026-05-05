from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey, Boolean, Numeric
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base

class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True)
    device_hash = Column(String(64), unique=True, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    scans = relationship("RawScan", back_populates="device")

class TowerPool(Base):
    """Stores Cell IDs verified by GPS-enabled 'Anchor' devices"""
    __tablename__ = "tower_pool"
    id = Column(Integer, primary_key=True)
    location_id = Column(String, index=True) # e.g., "AW212"
    cell_id = Column(Integer, index=True)
    last_confirmed_gps = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Integer, default=1)

class RawScan(Base):
    __tablename__ = "raw_scans"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=True)
    user_name = Column(String)
    location_id = Column(String) # Ground Truth
    wifi_data = Column(Text, nullable=True)
    
    # AI Predictions
    rf_prediction = Column(String)
    nn_prediction = Column(String)
    
    # Proximity Logic Columns
    proximity_verified = Column(Boolean, default=False)
    serving_cell = Column(Integer)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    cell_data = Column(JSON)
    gps_lat = Column(Numeric(precision=10, scale=6), nullable=True)
    gps_lon = Column(Numeric(precision=10, scale=6), nullable=True)
    
    device = relationship("Device", back_populates="scans")