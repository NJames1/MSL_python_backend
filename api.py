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
MIN_WIFI_RSSI = -85          
MIN_KNOWN_ROUTERS = 2        

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
        cell_info = payload.get('cellInfo', [])
        serving_cid = next((c.get('cid') for c in cell_info if c.get('isRegistered')), cell_info[0].get('cid') if cell_info else None)
        
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
            wifi_signals = {f"WIFI_{wifi['bssid']}": wifi['rssi'] for wifi in payload.get('wifiInfo', [])}
            cell_signals = {f"CELL_{cell['cid']}": cell['rsrp'] for cell in cell_info}
            signals = {**wifi_signals, **cell_signals}
            
            if signals:
                visible_known_routers = [k for k in wifi_signals.keys() if k in known_features_set]
                max_wifi = max(wifi_signals.values()) if wifi_signals else -100

                if len(visible_known_routers) < MIN_KNOWN_ROUTERS or max_wifi < MIN_WIFI_RSSI:
                    rf_res = "Outside AW"
                    nn_res = "Outside AW"
                else:
                    max_rssi = max(signals.values())
                    norm = {k: (v - max_rssi) for k, v in signals.items()}
                    input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                    
                    rf_prob = max(rf_model.predict_proba(input_df)[0])
                    raw_rf_pred = str(rf_model.predict(input_df)[0])

                    input_scaled = scaler.transform(input_df)
                    nn_prob = max(nn_model.predict_proba(input_scaled)[0])
                    raw_nn_pred = str(nn_model.predict(input_scaled)[0])

                    rf_res = raw_rf_pred if rf_prob >= CONFIDENCE_THRESHOLD else "Outside AW"
                    nn_res = raw_nn_pred if nn_prob >= CONFIDENCE_THRESHOLD else "Outside AW"

                    if raw_rf_pred == raw_nn_pred and ((rf_prob + nn_prob) / 2.0) >= CONSENSUS_THRESHOLD:
                        rf_res = nn_res = raw_rf_pred
                    elif nn_prob >= EXPERT_OVERRIDE_THRESHOLD:
                        rf_res = nn_res = raw_nn_pred
                    elif rf_prob >= EXPERT_OVERRIDE_THRESHOLD:
                        rf_res = nn_res = raw_rf_pred

        scan = models.RawScan(
            device_id=device.id, user_name=request.userName,
            location_id=request.locationId or "Unknown",
            rf_prediction=rf_res, nn_prediction=nn_res,
            proximity_verified=is_verified, serving_cell=serving_cid, 
            cell_data={"fingerprint": request.fingerprint}
        )
        db.add(scan); db.commit()

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