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
    model_features = joblib.load('model_features.pkl')
except Exception as e:
    logger.error(f"❌ Model load error: {e}")
    rf_model = nn_model = None

# --- OUT-OF-BOUNDS THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.45  # Must be at least 45% confident in a specific room
MIN_SIGNAL_STRENGTH = -85    # Strongest signal must be at least -85 dBm

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
        # 1. Handle Device
        device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
        if not device:
            device = models.Device(device_hash=request.deviceId); db.add(device); db.flush()

        # 2. ROBUST CID EXTRACTION
        payload = json.loads(request.fingerprint)
        cell_info = payload.get('cellInfo', [])
        serving_cid = None

        if cell_info:
            serving_cid = next((c.get('cid') for c in cell_info if c.get('isRegistered') == True), None)
            if serving_cid is None:
                serving_cid = cell_info[0].get('cid')
        
        logger.info(f"📡 Device {request.userName} latched to CID: {serving_cid}")

        # 3. ANCHOR LOGIC
        if request.gpsLat and request.gpsLon and request.locationId and serving_cid:
            anchor = db.query(models.TowerPool).filter_by(location_id=request.locationId, cell_id=serving_cid).first()
            if anchor:
                anchor.last_confirmed_gps = datetime.utcnow()
                anchor.confidence_score += 1
            else:
                db.add(models.TowerPool(location_id=request.locationId, cell_id=serving_cid))
            db.commit()

        # 4. PROXIMITY VERIFICATION
        is_verified = False
        if serving_cid and request.locationId:
            time_limit = datetime.utcnow() - timedelta(hours=5) 
            match = db.query(models.TowerPool).filter(
                models.TowerPool.location_id == request.locationId,
                models.TowerPool.cell_id == serving_cid,
                models.TowerPool.last_confirmed_gps >= time_limit
            ).first()
            is_verified = True if match else False

        # 5. INFERENCE (With OOD & Confidence Thresholding)
        rf_res = nn_res = "Unknown"
        
        if rf_model and nn_model and model_features:
            signals = {}
            for wifi in payload.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in cell_info:
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']
            
            if signals:
                max_rssi = max(signals.values())
                
                # OOD CHECK 1: Is the signal physically too weak?
                if max_rssi < MIN_SIGNAL_STRENGTH:
                    rf_res = "Outside AW"
                    nn_res = "Outside AW"
                else:
                    norm = {k: (v - max_rssi) for k, v in signals.items()}
                    input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                    
                    # OOD CHECK 2: Random Forest Probability
                    rf_prob = max(rf_model.predict_proba(input_df)[0])
                    if rf_prob >= CONFIDENCE_THRESHOLD:
                        rf_res = str(rf_model.predict(input_df)[0])
                    else:
                        rf_res = "Outside AW"

                    # OOD CHECK 3: Neural Network Probability
                    input_scaled = scaler.transform(input_df)
                    nn_prob = max(nn_model.predict_proba(input_scaled)[0])
                    if nn_prob >= CONFIDENCE_THRESHOLD:
                        nn_res = str(nn_model.predict(input_scaled)[0])
                    else:
                        nn_res = "Outside AW"

        # 6. SAVE RECORD 
        scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=request.locationId or "Unknown",
            rf_prediction=rf_res,
            nn_prediction=nn_res,
            proximity_verified=is_verified,
            serving_cell=serving_cid, 
            cell_data={"fingerprint": request.fingerprint},
            gps_lat=request.gpsLat,
            gps_lon=request.gpsLon
        )
        db.add(scan)
        db.commit()

        return {
            "success": True, 
            "proximity": "Verified" if is_verified else "Unverified",
            "cid": serving_cid
        }

    except Exception as e:
        db.rollback()
        logger.error(f"💥 Scan Error: {e}")
        return {"success": False, "error": str(e)}

# --- BATCH PROCESSING ENDPOINT ---
@router.post("/backfill-predictions")
def backfill_predictions(db: Session = Depends(get_db)):
    try:
        if not rf_model or not nn_model or not model_features:
            return {"success": False, "error": "Machine Learning models are not loaded into memory."}

        unlabelled_scans = db.query(models.RawScan).filter(
            (models.RawScan.rf_prediction == "Unknown") | 
            (models.RawScan.rf_prediction.is_(None)) |
            (models.RawScan.nn_prediction == "Unknown") |
            (models.RawScan.nn_prediction.is_(None))
        ).all()

        if not unlabelled_scans:
            return {"success": True, "message": "No unlabelled scans found.", "processed": 0}

        processed_count = 0
        
        for scan in unlabelled_scans:
            try:
                cell_data = scan.cell_data
                if isinstance(cell_data, str):
                    cell_data = json.loads(cell_data)
                    
                fingerprint_str = cell_data.get('fingerprint', '{}')
                payload = json.loads(fingerprint_str)

                signals = {}
                for wifi in payload.get('wifiInfo', []):
                    signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
                for cell in payload.get('cellInfo', []):
                    signals[f"CELL_{cell['cid']}"] = cell['rsrp']

                if signals:
                    max_rssi = max(signals.values())
                    
                    # Apply identical OOD Logic to backfill
                    if max_rssi < MIN_SIGNAL_STRENGTH:
                        scan.rf_prediction = "Outside AW"
                        scan.nn_prediction = "Outside AW"
                    else:
                        norm = {k: (v - max_rssi) for k, v in signals.items()}
                        input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                        
                        rf_prob = max(rf_model.predict_proba(input_df)[0])
                        scan.rf_prediction = str(rf_model.predict(input_df)[0]) if rf_prob >= CONFIDENCE_THRESHOLD else "Outside AW"
                        
                        input_scaled = scaler.transform(input_df)
                        nn_prob = max(nn_model.predict_proba(input_scaled)[0])
                        scan.nn_prediction = str(nn_model.predict(input_scaled)[0]) if nn_prob >= CONFIDENCE_THRESHOLD else "Outside AW"
                        
                    processed_count += 1
                    
            except Exception as row_error:
                logger.error(f"⚠️ Error processing scan ID {scan.id}: {row_error}")
                continue 

        db.commit()
        return {
            "success": True, 
            "message": f"Successfully updated locations for {processed_count} unlabelled scans.",
            "processed": processed_count
        }

    except Exception as e:
        db.rollback()
        logger.error(f"💥 Backfill Error: {e}")
        return {"success": False, "error": str(e)}