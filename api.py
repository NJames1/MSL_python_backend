import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db, SessionLocal
import models
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Import your Machine Learning matching logic
from matching import predict_device_location

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Pydantic Models for Request/Response Validation ---

class DashboardStatsResponse(BaseModel):
    totalDevices: int
    presentDevices: int
    absentDevices: int
    averageConfidence: float
    recentScans: List[dict]

class ScanResultResponse(BaseModel):
    id: int
    deviceId: str
    confidence: float
    locationId: Optional[str] = None
    timestamp: str
    matched: bool
    lat: Optional[float] = None
    lng: Optional[float] = None

class ScanSubmitRequest(BaseModel):
    deviceId: str
    userName: str        
    fingerprint: str
    locationId: Optional[str] = None # The server receives the Room Name here
    gpsLat: Optional[float] = None
    gpsLon: Optional[float] = None

class ApiResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


# --- API Endpoints ---

@router.get("/stats", response_model=DashboardStatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    try:
        total_devices = db.query(models.Device).count()
        
        recent_scans = db.query(models.RawScan).order_by(
            models.RawScan.timestamp.desc()
        ).limit(50).all()
        
        fingerprints = db.query(models.Fingerprint).all()
        avg_confidence = 0.0
        if fingerprints:
            confidences = [float(fp.confidence) for fp in fingerprints if fp.confidence]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        recent_scans_formatted = [
            {
                "id": f"scan-{scan.id}",
                "deviceId": f"DEV{scan.device_id or 0:03d}",
                "confidence": 0.85,
                "locationId": scan.location_id, # Updated to show room in stats
                "timestamp": scan.timestamp.isoformat() if scan.timestamp else datetime.utcnow().isoformat(),
                "matched": True,
                "lat": scan.gps_lat,
                "lng": scan.gps_lon
            }
            for scan in recent_scans[:5]
        ]
        
        response = DashboardStatsResponse(
            totalDevices=total_devices,
            presentDevices=max(1, total_devices // 2),
            absentDevices=max(0, total_devices - (total_devices // 2)),
            averageConfidence=min(avg_confidence, 0.95),
            recentScans=recent_scans_formatted
        )
        return response
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return DashboardStatsResponse(totalDevices=0, presentDevices=0, absentDevices=0, averageConfidence=0.0, recentScans=[])

@router.get("/live", response_model=List[ScanResultResponse])
def get_live_scans(db: Session = Depends(get_db)):
    """Get live scan results with Machine Learning Inference applied"""
    try:
        scans = db.query(models.RawScan).order_by(
            models.RawScan.timestamp.desc()
        ).limit(20).all()
        
        results = []
        for scan in scans:
            predicted_lat, predicted_lon, ml_confidence = predict_device_location(scan.cell_data, db)
            
            results.append(
                ScanResultResponse(
                    id=scan.id,
                    deviceId=f"DEV{scan.device_id or 0:03d}",
                    confidence=ml_confidence,
                    locationId=scan.location_id, # Display ground truth if available
                    timestamp=scan.timestamp.isoformat() if scan.timestamp else datetime.utcnow().isoformat(),
                    matched=True if ml_confidence > 0.0 else False,
                    lat=predicted_lat,
                    lng=predicted_lon
                )
            )
        return results
    except Exception as e:
        logger.error(f"Error getting live scans: {str(e)}")
        return []

@router.post("/scan", response_model=ApiResponse)
def submit_scan(request: ScanSubmitRequest, db: Session = Depends(get_db)):
    """Submit a new scan payload from the Android Client"""
    try:
        # Find or create device profile
        device = db.query(models.Device).filter(
            models.Device.device_hash == request.deviceId
        ).first()
        
        if not device:
            device = models.Device(device_hash=request.deviceId)
            db.add(device)
            db.flush()
        
        # --- THE CRITICAL FIX ---
        # We now pass location_id=request.locationId to the DB model
        scan = models.RawScan(
            device_id=device.id,
            user_name=request.userName,
            location_id=request.locationId,  # <--- SAVES ROOM NAME (e.g., AW201)
            cell_data={"fingerprint": request.fingerprint}, 
            wifi_data={"submitted": True},
            gps_lat=request.gpsLat if request.gpsLat is not None else 0.0,
            gps_lon=request.gpsLon if request.gpsLon is not None else 0.0
        )
        db.add(scan)
        db.commit()
        
        logger.info(f"Scan submitted successfully: Device {request.deviceId} at {request.locationId}")
        return ApiResponse(
            success=True,
            data={
                "scanId": scan.id,
                "location": request.locationId,
                "timestamp": scan.timestamp.isoformat()
            }
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error submitting scan: {str(e)}")
        return ApiResponse(success=False, error=str(e))

@router.get("/health")
def health_check():
    return {"status": "healthy", "message": "MSL Backend is running"}