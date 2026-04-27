import logging
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from db import get_db
import models

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- 2. LOAD MACHINE LEARNING ARTIFACTS ---
try:
    rf_model = joblib.load('localization_model.pkl')  # Your 93.47% RF Model
    nn_model = joblib.load('nn_model.pkl')            # New Small Neural Network
    scaler = joblib.load('scaler.pkl')                # Z-score Scaler for NN
    model_features = joblib.load('model_features.pkl')# Global Feature Map
    logger.info("✅ All ML Models and Scalers loaded successfully.")
except Exception as e:
    logger.error(f"❌ ML Initialization Failed: {e}")
    rf_model = nn_model = scaler = model_features = None

# --- 3. DATA MODELS (PYDANTIC) ---
class ScanSubmitRequest(BaseModel):
    deviceId: str
    userName: str        
    fingerprint: str
    locationId: Optional[str] = None # The "Ground Truth" room selected in app
    gpsLat: Optional[float] = None   # Provided only by "Anchor" devices
    gpsLon: Optional[float] = None

class ApiResponse(BaseModel):
    success: bool
    proximity: str
    rf_room: str
    nn_room: str
    message: str

# --- 4. CORE LOGIC ENDPOINT ---

@router.post("/scan", response_model=ApiResponse)
def submit_scan(request: ScanSubmitRequest, db: Session = Depends(get_db)):
    """
    Handles Real-Time Localization, Parallel Inference, and Collaborative Proximity.
    """
    try:
        # A. Handle Device Registration
        device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
        if not device:
            device = models.Device(device_hash=request.deviceId)
            db.add(device)
            db.flush()

        # B. Parse Fingerprint for Cell CID (Proximity Key)
        payload = json.loads(request.fingerprint)
        serving_cid = None
        for cell in payload.get('cellInfo', []):
            if cell.get('isRegistered'):
                serving_cid = cell.get('cid')
                break

        # C. COLLABORATIVE ANCHOR LOGIC (Lecturer / GPS Case)
        # If the device has GPS on, it "vets" the Cell Tower for this room.
        if request.gpsLat and request.gpsLon and request.locationId and serving_cid:
            anchor = db.query(models.TowerPool).filter_by(
                location_id=request.locationId, 
                cell_id=serving_cid
            ).first()
            
            if anchor:
                anchor.last_confirmed_gps = datetime.utcnow()
                anchor.confidence_score += 1
            else:
                new_anchor = models.TowerPool(
                    location_id=request.locationId,
                    cell_id=serving_cid
                )
                db.add(new_anchor)
            db.commit()
            logger.info(f"⚓ Anchor Updated: Room {request.locationId} <-> Cell {serving_cid}")

        # D. PROXIMITY VALIDATION (Student / No-GPS Case)
        is_verified = False
        if serving_cid and request.locationId:
            # Look for a vetted anchor in this room from the last 2 hours
            time_limit = datetime.utcnow() - timedelta(hours=2)
            match = db.query(models.TowerPool).filter(
                models.TowerPool.location_id == request.locationId,
                models.TowerPool.cell_id == serving_cid,
                models.TowerPool.last_confirmed_gps >= time_limit
            ).first()
            is_verified = True if match else False

        # E. PARALLEL INFERENCE ENGINE (RF + NN)
        rf_pred = "Inference Error"
        nn_pred = "Inference Error"

        if rf_model and nn_model and model_features:
            # 1. Feature Extraction (Differential RSSI)
            signals = {}
            for wifi in payload.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in payload.get('cellInfo', []):
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']

            if signals:
                max_rssi = max(signals.values())
                diff_features = {k: (v - max_rssi) for k, v in signals.items()}
                
                # Align with training columns
                input_df = pd.DataFrame([diff_features], columns=model_features).fillna(-100)

                # 2. Random Forest Prediction
                rf_pred = str(rf_model.predict(input_df)[0])

                # 3. Neural Network Prediction (Requires Scaling)
                input_scaled = scaler.transform(input_df)
                nn_pred = str(nn_model.predict(input_scaled)[0])

        # F. SAVE RESULTS TO DATABASE
        new_scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=request.locationId or "Unknown",
            rf_prediction=rf_pred,
            nn_prediction=nn_pred,
            proximity_verified=is_verified,
            serving_cell=serving_cid,
            cell_data={"fingerprint": request.fingerprint},
            gps_lat=request.gpsLat,
            gps_lon=request.gpsLon
        )
        db.add(new_scan)
        db.commit()

        # G. CONSTRUCT RESPONSE
        prox_text = "Verified" if is_verified else "Unverified/Outside Pool"
        return ApiResponse(
            success=True,
            proximity=prox_text,
            rf_room=rf_pred,
            nn_room=nn_pred,
            message=f"Results for {request.userName}: RF={rf_pred}, NN={nn_pred}, Prox={prox_text}"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"💥 Critical Error in /scan: {e}")
        return ApiResponse(
            success=False,
            proximity="Error",
            rf_room="Error",
            nn_room="Error",
            message=str(e)
        )

# --- 5. UTILITY ENDPOINTS ---

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "Parallel RF + SNN",
        "proximity_layer": "Active",
        "models_loaded": rf_model is not None
    }