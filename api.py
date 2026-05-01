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

        # 2. ROBUST CID EXTRACTION (The Fix)
        payload = json.loads(request.fingerprint)
        cell_info = payload.get('cellInfo', [])
        serving_cid = None

        if cell_info:
            # Step A: Look for the tower explicitly marked as "registered" (active latch)
            serving_cid = next((c.get('cid') for c in cell_info if c.get('isRegistered') == True), None)
            
            # Step B: Fallback—if none are marked registered, take the first one in the list
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
            time_limit = datetime.utcnow() - timedelta(hours=5) # Increased window for demo
            match = db.query(models.TowerPool).filter(
                models.TowerPool.location_id == request.locationId,
                models.TowerPool.cell_id == serving_cid,
                models.TowerPool.last_confirmed_gps >= time_limit
            ).first()
            is_verified = True if match else False

        # 5. INFERENCE (Differential RSSI)
        rf_res = nn_res = "Unknown"
        if rf_model and nn_model and model_features:
            signals = {}
            for wifi in payload.get('wifiInfo', []):
                signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
            for cell in cell_info:
                signals[f"CELL_{cell['cid']}"] = cell['rsrp']
            
            if signals:
                max_rssi = max(signals.values())
                norm = {k: (v - max_rssi) for k, v in signals.items()}
                input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                rf_res = str(rf_model.predict(input_df)[0])
                nn_res = str(nn_model.predict(scaler.transform(input_df))[0])

        # 6. SAVE RECORD (Explicitly including serving_cell)
        scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=request.locationId or "Unknown",
            rf_prediction=rf_res,
            nn_prediction=nn_res,
            proximity_verified=is_verified,
            serving_cell=serving_cid,  # <--- THIS IS NOW MAPPED
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

# --- NEW BATCH PROCESSING ENDPOINT ---
@router.post("/backfill-predictions")
def backfill_predictions(db: Session = Depends(get_db)):
    """
    Finds all records in the database with missing predictions ('Unknown' or null),
    runs the ML inference on their raw cell_data, and updates the database.
    """
    try:
        if not rf_model or not nn_model or not model_features:
            return {"success": False, "error": "Machine Learning models are not loaded into memory."}

        # Query all rows where predictions are missing
        unlabelled_scans = db.query(models.RawScan).filter(
            (models.RawScan.rf_prediction == "Unknown") | 
            (models.RawScan.rf_prediction.is_(None)) |
            (models.RawScan.nn_prediction == "Unknown") |
            (models.RawScan.nn_prediction.is_(None))
        ).all()

        if not unlabelled_scans:
            return {"success": True, "message": "No unlabelled scans found. Database is up to date.", "processed": 0}

        processed_count = 0
        
        for scan in unlabelled_scans:
            try:
                # 1. Safely extract JSON fingerprint
                cell_data = scan.cell_data
                if isinstance(cell_data, str):
                    cell_data = json.loads(cell_data)
                    
                fingerprint_str = cell_data.get('fingerprint', '{}')
                payload = json.loads(fingerprint_str)

                # 2. Extract signals
                signals = {}
                for wifi in payload.get('wifiInfo', []):
                    signals[f"WIFI_{wifi['bssid']}"] = wifi['rssi']
                for cell in payload.get('cellInfo', []):
                    signals[f"CELL_{cell['cid']}"] = cell['rsrp']

                # 3. Apply Differential RSSI & Predict
                if signals:
                    max_rssi = max(signals.values())
                    norm = {k: (v - max_rssi) for k, v in signals.items()}
                    
                    # Align to model features
                    input_df = pd.DataFrame([norm], columns=model_features).fillna(-100)
                    
                    # Execute Parallel Inference
                    scan.rf_prediction = str(rf_model.predict(input_df)[0])
                    scan.nn_prediction = str(nn_model.predict(scaler.transform(input_df))[0])
                    processed_count += 1
                    
            except Exception as row_error:
                logger.error(f"⚠️ Error processing scan ID {scan.id}: {row_error}")
                continue # Skip failing rows and move to the next

        # Commit all updated predictions to the database
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