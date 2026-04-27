import logging
import json
import joblib
import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
import models
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta

router = APIRouter()

# Load ML Models
try:
    rf_model = joblib.load('localization_model.pkl')
    nn_model = joblib.load('nn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')
except Exception as e:
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
    device = db.query(models.Device).filter(models.Device.device_hash == request.deviceId).first()
    if not device:
        device = models.Device(device_hash=request.deviceId); db.add(device); db.flush()

    # 1. Extract Cell ID
    payload = json.loads(request.fingerprint)
    serving_cid = next((c.get('cid') for c in payload.get('cellInfo', []) if c.get('isRegistered')), None)

    # 2. Collaborative Anchor Logic (Lecturer/GPS Case)
    if request.gpsLat and request.gpsLon and request.locationId and serving_cid:
        anchor = db.query(models.TowerPool).filter_by(location_id=request.locationId, cell_id=serving_cid).first()
        if anchor:
            anchor.last_confirmed_gps = datetime.utcnow()
            anchor.confidence_score += 1
        else:
            db.add(models.TowerPool(location_id=request.locationId, cell_id=serving_cid))
        db.commit()

    # 3. Proximity Validation (Student/No-GPS Case)
    is_verified = False
    if serving_cid and request.locationId:
        recent_threshold = datetime.utcnow() - timedelta(hours=2)
        match = db.query(models.TowerPool).filter(
            models.TowerPool.location_id == request.locationId,
            models.TowerPool.cell_id == serving_cid,
            models.TowerPool.last_confirmed_gps >= recent_threshold
        ).first()
        is_verified = True if match else False

    # 4. Run Parallel Inference (RF + NN)
    rf_res = nn_res = "Unknown"
    if rf_model and nn_model:
        # (Assuming Differential RSSI processing logic here as previously defined)
        # ... logic to generate input_df and input_scaled ...
        rf_res = str(rf_model.predict(input_df)[0])
        nn_res = str(nn_model.predict(scaler.transform(input_df))[0])

    # 5. Save and Return
    scan = models.RawScan(
        device_id=device.id, user_name=request.userName,
        location_id=request.locationId, rf_prediction=rf_res,
        nn_prediction=nn_res, proximity_verified=is_verified,
        serving_cell=serving_cid, cell_data={"fingerprint": request.fingerprint}
    )
    db.add(scan); db.commit()
    
    return {"success": True, "proximity": "Verified" if is_verified else "Unverified", "rf": rf_res}