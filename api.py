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

# --- 1. LOAD MODEL ARTIFACTS ---
# We load both the Random Forest and the Neural Network + its Scaler
try:
    # Existing RF Model
    rf_model = joblib.load('localization_model.pkl')
    # New Neural Network Model
    nn_model = joblib.load('nn_model.pkl')
    # Essential Scaler for the Neural Network
    scaler = joblib.load('scaler.pkl')
    # Shared Feature Map
    model_features = joblib.load('model_features.pkl')
    
    logger.info("✅ RF and NN Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load ML models: {e}")
    rf_model = None
    nn_model = None

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
    """Receive scan, predict via RF and NN, and store results"""
    try:
        device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
        if not device:
            device = models.Device(device_hash=request.deviceId)
            db.add(device)
            db.flush()

        # Placeholders
        rf_prediction = "Unknown"
        nn_prediction = "Unknown"
        ground_truth = request.locationId or "Unknown Area"
        
        if rf_model and nn_model and request.fingerprint:
            try:
                # 1. Parse and Extract Signals
                payload = json.loads(request.fingerprint)
                signals = {}
                for wifi in payload.get('wifiInfo', []):
                    signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
                for cell in payload.get('cellInfo', []):
                    signals[f"CELL_{cell['cid']}"] = cell['rsrp']

                if signals:
                    # 2. Hardware-Agnostic Normalization (Differential RSSI)
                    max_rssi = max(signals.values())
                    diff_features = {k: (v - max_rssi) for k, v in signals.items()}
                    input_df = pd.DataFrame([diff_features], columns=model_features).fillna(-100)

                    # 3. Random Forest Inference (Direct)
                    rf_res = rf_model.predict(input_df)[0]
                    rf_prediction = str(rf_res)

                    # 4. Neural Network Inference (Requires Z-score Scaling)
                    input_scaled = scaler.transform(input_df)
                    nn_res = nn_model.predict(input_scaled)[0]
                    nn_prediction = str(nn_res)

            except Exception as e:
                logger.error(f"Inference error: {e}")

        # Save scan with BOTH predictions for Chapter 4 comparison
        scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=ground_truth, # Store the room user actually selected
            rf_prediction=rf_prediction,
            nn_prediction=nn_prediction,
            cell_data={"fingerprint": request.fingerprint}, 
            gps_lat=request.gpsLat or 0.0,
            gps_lon=request.gpsLon or 0.0
        )
        db.add(scan)
        db.commit()

        return ApiResponse(
            success=True,
            data={
                "message": f"RF: {rf_prediction} | NN: {nn_prediction}",
                "rf_location": rf_prediction,
                "nn_location": nn_prediction,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        db.rollback()
        return ApiResponse(success=False, error=str(e))

# --- 3. SYSTEM HEALTH & STATS ---

@router.get("/health")
def health_check():
    return {"status": "healthy", "message": "Parallel Inference Backend Active"}

@router.get("/stats", response_model=DashboardStatsResponse)
def get_stats(db: Session = Depends(get_db)):
    total_devices = db.query(models.Device).count()
    recent_scans = db.query(models.RawScan).order_by(models.RawScan.timestamp.desc()).limit(10).all()
    
    return DashboardStatsResponse(
        totalDevices=total_devices,
        presentDevices=total_devices,
        absentDevices=0,
        averageConfidence=0.93,
        recentScans=[{
            "id": s.id, 
            "truth": s.location_id, 
            "rf": s.rf_prediction, 
            "nn": s.nn_prediction
        } for s in recent_scans]
    )