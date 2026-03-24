import numpy as np

RSSI_MIN = -120
RSSI_MAX = -30

def normalize(value):
    return max(RSSI_MIN, min(RSSI_MAX, value))


def build_feature_vector(scan):
    features = {}
    for cell in scan['cell_info']:
        features[f"cell_{cell['cid']}"] = normalize(cell['rsrp'])
    for wifi in scan['wifi_info']:
        features[f"wifi_{wifi['bssid']}"] = normalize(wifi['rssi'])
    return features
