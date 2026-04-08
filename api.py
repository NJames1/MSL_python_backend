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
    fingerprint: str
    locationId: Optional[str] = None
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
        # Count total registered devices
        total_devices = db.query(models.Device).count()
        
        # Fetch the 50 most recent raw scans
        recent_scans = db.query(models.RawScan).order_by(
            models.RawScan.timestamp.desc()
        ).limit(50).all()
        
        # Calculate average confidence from the calibrated Fingerprint database
        fingerprints = db.query(models.Fingerprint).all()
        avg_confidence = 0.0
        if fingerprints:
            confidences = [float(fp.confidence) for fp in fingerprints if fp.confidence]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        # Format recent scans for the dashboard table
        recent_scans_formatted = [
            {
                "id": f"scan-{scan.id}",
                "deviceId": f"DEV{scan.device_id or 0:03d}",
                "confidence": 0.85, # Default placeholder for raw scan list
                "locationId": None,
                "timestamp": scan.timestamp.isoformat() if scan.timestamp else datetime.utcnow().isoformat(),
                "matched": True,
                "lat": scan.gps_lat,
                "lng": scan.gps_lon
            }
            for scan in recent_scans[:5]
        ]
        
        response = DashboardStatsResponse(
            totalDevices=total_devices,
            presentDevices=max(1, total_devices // 2),  # Estimation logic
            absentDevices=max(0, total_devices - (total_devices // 2)),
            averageConfidence=min(avg_confidence, 0.95),
            recentScans=recent_scans_formatted
        )
        logger.info("Dashboard stats retrieved")
        return response
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        # Return mock data on error to prevent dashboard crash
        return DashboardStatsResponse(
            totalDevices=0,
            presentDevices=0,
            absentDevices=0,
            averageConfidence=0.0,
            recentScans=[]
        )

@router.get("/live", response_model=List[ScanResultResponse])
def get_live_scans(db: Session = Depends(get_db)):
    """Get live scan results with Machine Learning Inference applied"""
    try:
        # Pull the latest 20 scans from the database
        scans = db.query(models.RawScan).order_by(
            models.RawScan.timestamp.desc()
        ).limit(20).all()
        
        results = []
        for scan in scans:
            # PHASE 2: MACHINE LEARNING INFERENCE
            # Pass the raw radio signals to the matching algorithm to predict the true location
            predicted_lat, predicted_lon, ml_confidence = predict_device_location(scan.cell_data, db)
            
            results.append(
                ScanResultResponse(
                    id=scan.id,
                    deviceId=f"DEV{scan.device_id or 0:03d}",
                    confidence=ml_confidence, # Use the actual ML confidence score
                    locationId=None,
                    timestamp=scan.timestamp.isoformat() if scan.timestamp else datetime.utcnow().isoformat(),
                    matched=True if ml_confidence > 0.0 else False,
                    lat=predicted_lat, # Map uses predicted coordinates, not raw GPS
                    lng=predicted_lon  # Map uses predicted coordinates, not raw GPS
                )
            )
            
        logger.info(f"Retrieved and processed {len(results)} live scans via ML")
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
        
        # Create raw scan record with fallback coordinates if GPS is unavailable
        scan = models.RawScan(
            device_id=device.id,
            cell_data={"fingerprint": request.fingerprint}, 
            wifi_data={"submitted": True},
            gps_lat=request.gpsLat if request.gpsLat is not None else 0.0,
            gps_lon=request.gpsLon if request.gpsLon is not None else 0.0
        )
        db.add(scan)
        db.commit()
        
        logger.info(f"Scan submitted successfully for device {request.deviceId}")
        return ApiResponse(
            success=True,
            data={
                "scanId": scan.id,
                "deviceId": request.deviceId,
                "timestamp": scan.timestamp.isoformat()
            }
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error submitting scan: {str(e)}")
        return ApiResponse(
            success=False,
            error=str(e)
        )

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MSL Backend is running"}