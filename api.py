import logging
import json
import joblib
import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

from db import get_db
import models

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- LOAD MODELS ---
try:
    rf_model = joblib.load('localization_model.pkl')
    nn_model = joblib.load('nn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = list(joblib.load('model_features.pkl'))
    known_features_set = set(model_features)
    logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
except Exception as e:
    logger.error(f"❌ CRITICAL: Model load error: {e}")
    rf_model = nn_model = model_features = known_features_set = None

# --- THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.55
CONSENSUS_THRESHOLD = 0.45
EXPERT_OVERRIDE_THRESHOLD = 0.60
# Note: Wi-Fi thresholds are no longer used for power-optimized cell-only tracking

class ScanSubmitRequest(BaseModel):
    deviceId: str
    userName: str        
    fingerprint: str
    locationId: Optional[str] = None 
    gpsLat: Optional[float] = None
    gpsLon: Optional[float] = None

@router.post("/scan")
def submit_scan(request: ScanSubmitRequest, db: Session = Depends(get_db)):
    try:
        device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
        if not device:
            device = models.Device(device_hash=request.deviceId); db.add(device); db.flush()

        payload = json.loads(request.fingerprint)
        # Handle the new data structure from Android (cell_data instead of cellInfo)
        cell_info = payload.get('cell_data', payload.get('cellInfo', []))
        
        # Safely extract CID (some modems use 'cellId', older versions used 'cid')
        serving_cid = None
        if cell_info:
            first_cell = cell_info[0]
            serving_cid = first_cell.get('cellId', first_cell.get('cid'))
        
        is_verified = False
        if serving_cid and request.locationId:
            time_limit = datetime.utcnow() - timedelta(hours=5) 
            match = db.query(models.TowerPool).filter(
                models.TowerPool.location_id == request.locationId,
                models.TowerPool.cell_id == serving_cid,
                models.TowerPool.last_confirmed_gps >= time_limit
            ).first()
            is_verified = True if match else False

        rf_res = nn_res = "Unknown"
        
        if rf_model and model_features:
            # 1. POWER OPTIMIZATION: Process Cell Data Only
            cell_signals = {f"CELL_{cell.get('cellId', cell.get('cid'))}": cell.get('rssi', cell.get('rsrp')) for cell in cell_info}
            signals = cell_signals
            
            if signals:
                # 2. Skip Wi-Fi checks and proceed directly to normalization
                max_rssi = max(signals.values())
                norm = {k: (v - max_rssi) for k, v in signals.items()}
                
                # .fillna(-100) automatically injects weak signals for the missing Wi-Fi features
                # so your pre-trained model doesn't crash from missing columns!
                input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                
                # 3. Random Forest Inference
                rf_prob = max(rf_model.predict_proba(input_df)[0])
                raw_rf_pred = str(rf_model.predict(input_df)[0])

                # 4. Small Neural Network Inference
                input_scaled = scaler.transform(input_df)
                nn_prob = max(nn_model.predict_proba(input_scaled)[0])
                raw_nn_pred = str(nn_model.predict(input_scaled)[0])

                # 5. Confidence checks
                rf_res = raw_rf_pred if rf_prob >= CONFIDENCE_THRESHOLD else "Outside AW"
                nn_res = raw_nn_pred if nn_prob >= CONFIDENCE_THRESHOLD else "Outside AW"

                # 6. Consensus Engine
                if raw_rf_pred == raw_nn_pred and ((rf_prob + nn_prob) / 2.0) >= CONSENSUS_THRESHOLD:
                    rf_res = nn_res = raw_rf_pred
                elif nn_prob >= EXPERT_OVERRIDE_THRESHOLD:
                    rf_res = nn_res = raw_nn_pred
                elif rf_prob >= EXPERT_OVERRIDE_THRESHOLD:
                    rf_res = nn_res = raw_rf_pred
        
        # 7. SAVE TO DATABASE (Wi-Fi data removed to prevent RawScan crash)
        scan = models.RawScan(
            device_id=device.id, 
            user_name=request.userName,
            location_id=request.locationId or "Unknown",
            rf_prediction=rf_res, 
            nn_prediction=nn_res,
            proximity_verified=is_verified, 
            serving_cell=serving_cid, 
            cell_data={"fingerprint": request.fingerprint}
        )
        db.add(scan)
        db.commit()

        # --- ALIGNED RETURN BLOCK ---
        logger.info(f"✅ Prediction for {request.userName}: {rf_res}")
        return {
            "success": True, 
            "data": {
                "message": rf_res,      # Matches MainActivity result parsing
                "proximity": "Verified" if is_verified else "Unverified",
                "nn_prediction": nn_res
            }
        }

    except Exception as e:
        db.rollback()
        logger.error(f"💥 Scan Error: {e}")
        return {"success": False, "error": str(e)}