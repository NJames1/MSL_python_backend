from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey, Numeric
from datetime import datetime
from db import Base


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True)
    device_hash = Column(String(64), unique=True, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)


class RawScan(Base):
    __tablename__ = "raw_scans"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cell_data = Column(JSON)
    wifi_data = Column(JSON)
    gps_lat = Column(Numeric(precision=10, scale=6), nullable=True)
    gps_lon = Column(Numeric(precision=10, scale=6), nullable=True)


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
