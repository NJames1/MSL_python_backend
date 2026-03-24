import numpy as np


def cosine_similarity(v1, v2):
    keys = set(v1.keys()).union(v2.keys())
    a = np.array([v1.get(k, -120) for k in keys])
    b = np.array([v2.get(k, -120) for k in keys])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Submodule test change: timestamp Sat 14 Feb 2026 02:03:55 AM EAT
