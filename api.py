import logging
import json
import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db import get_db
import models
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# --- 1. LOAD THE TRAINED MODEL ---
# Ensure these .pkl files are in your root folder alongside app.py
try:
    model = joblib.load('localization_model.pkl')
    model_features = joblib.load('model_features.pkl')
    logger.info("✅ Random Forest Model and Feature Map loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load ML model: {e}")
    model = None

# --- Pydantic Models for Validation ---
class ApiResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

class ScanSubmitRequest(BaseModel):
    deviceId: str
    userName: str        
    fingerprint: str
    locationId: Optional[str] = None 
    gpsLat: Optional[float] = None
    gpsLon: Optional[float] = None

class DashboardStatsResponse(BaseModel):
    totalDevices: int
    presentDevices: int
    absentDevices: int
    averageConfidence: float
    recentScans: List[dict]

# --- 2. LOCALIZATION ENDPOINT ---

@router.post("/scan", response_model=ApiResponse)
def submit_scan(request: ScanSubmitRequest, db: Session = Depends(get_db)):
    """Receive scan, predict location via Random Forest, and return result"""
    try:
        device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
        if not device:
            device = models.Device(device_hash=request.deviceId)
            db.add(device)
            db.flush()

        # Default if inference fails
        predicted_room = request.locationId or "Unknown Area"
        
        if model and request.fingerprint:
            try:
                payload = json.loads(request.fingerprint)
                signals = {}
                
                # Extract signals for the Differential RSSI Algorithm
                for wifi in payload.get('wifiInfo', []):
                    signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
                for cell in payload.get('cellInfo', []):
                    signals[f"CELL_{cell['cid']}"] = cell['rsrp']

                if signals:
                    # Apply hardware-agnostic normalization
                    max_rssi = max(signals.values())
                    diff_features = {k: (v - max_rssi) for k, v in signals.items()}

                    # Align with 93.47% accuracy training features
                    input_df = pd.DataFrame([diff_features], columns=model_features).fillna(-100)
                    prediction = model.predict(input_df)[0]
                    predicted_room = str(prediction)
            except Exception as e:
                logger.error(f"ML Inference error: {e}")

        # Save scan with the PREDICTED location
        scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=predicted_room, 
            cell_data={"fingerprint": request.fingerprint}, 
            gps_lat=request.gpsLat or 0.0,
            gps_lon=request.gpsLon or 0.0
        )
        db.add(scan)
        db.commit()

        return ApiResponse(
            success=True,
            data={
                "message": f"You are in {predicted_room}",
                "location": predicted_room,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        db.rollback()
        return ApiResponse(success=False, error=str(e))

# --- 3. SYSTEM HEALTH & STATS ---

@router.get("/health")
def health_check():
    """Endpoint for system verification"""
    return {"status": "healthy", "message": "Power-Optimized Backend is active"}

@router.get("/stats", response_model=DashboardStatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Retrieve statistics for the administrative dashboard"""
    total_devices = db.query(models.Device).count()
    recent_scans = db.query(models.RawScan).order_by(models.RawScan.timestamp.desc()).limit(5).all()
    
    return DashboardStatsResponse(
        totalDevices=total_devices,
        presentDevices=total_devices, # Placeholder for demo
        absentDevices=0,
        averageConfidence=0.93, # Reflects your 93.47% accuracy
        recentScans=[{"id": s.id, "location": s.location_id} for s in recent_scans]
    )