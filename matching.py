import numpy as np
import models # Imports your database models

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two signal dictionaries."""
    if not v1 or not v2:
        return 0.0
        
    keys = set(v1.keys()).union(v2.keys())
    # -120 dBm acts as the baseline for a completely missing signal
    a = np.array([v1.get(k, -120) for k in keys])
    b = np.array([v2.get(k, -120) for k in keys])
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))

def predict_device_location(live_scan_dict, db):
    """
    Takes a live scan dictionary, compares it to all stored fingerprints,
    and returns the best matching coordinates and confidence score.
    """
    # 1. Fetch all stored fingerprints (The Radio Map) from the database
    fingerprints = db.query(models.Fingerprint).all()
    
    best_match_location_id = None
    highest_similarity = -1.0 # Cosine similarity ranges from -1 to 1

    # 2. Loop through the database and compare the live scan to every fingerprint
    for fp in fingerprints:
        # fp.features is the stored dictionary of signals for that Anchor Point
        score = cosine_similarity(live_scan_dict, fp.features)
        
        # If this is the closest match we've seen so far, remember it
        if score > highest_similarity:
            highest_similarity = score
            best_match_location_id = fp.location_id

    # 3. If we found a good match (e.g., above an 85% confidence threshold)
    if best_match_location_id and highest_similarity > 0.85:
        # Look up the actual latitude and longitude of that Anchor Point
        location = db.query(models.Location).filter(models.Location.id == best_match_location_id).first()
        if location:
            return location.centroid_lat, location.centroid_lon, highest_similarity
            
    # 4. Fallback if no match is found (User is somewhere unmapped)
    return 0.0, 0.0, highest_similarity if highest_similarity > -1.0 else 0.0

# Submodule test change: timestamp Sat 14 Feb 2026 02:03:55 AM EAT