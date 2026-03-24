from db import SessionLocal, init_db
import models


def seed():
    # ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        # check if any device exists
        existing = db.query(models.Device).first()
        if existing:
            print("Seed data already present — skipping.")
            return

        # create a sample device
        device = models.Device(device_hash="test-device-1")
        db.add(device)
        db.flush()

        # create a sample location
        loc = models.Location(centroid_lat=37.7749, centroid_lon=-122.4194)
        db.add(loc)
        db.flush()

        # create a fingerprint for that location
        fp = models.Fingerprint(location_id=loc.id, features={"sample": "value"}, confidence=0.9000)
        db.add(fp)

        # add a raw scan pointing to device
        scan = models.RawScan(device_id=device.id, cell_data={"cells": []}, wifi_data={"aps": []}, gps_lat=37.7749, gps_lon=-122.4194)
        db.add(scan)

        db.commit()
        print("Seed data inserted.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    seed()
