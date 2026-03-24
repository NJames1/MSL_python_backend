from matching import cosine_similarity


SIMILARITY_THRESHOLD = 0.85


def match_fingerprint(features, fingerprints):
    best = None
    score = 0
    for fp in fingerprints:
        s = cosine_similarity(features, fp.features)
        if s > score:
            score = s
            best = fp
    return best, score