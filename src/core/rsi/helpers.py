"""Helpers kept simple - SRP: only y extraction."""
def hip_center_y(kps):
    if kps is None: return None
    lh, rh = kps.get("left_hip"), kps.get("right_hip")
    if lh is None or rh is None: return None
    return float((lh["y"] + rh["y"]) / 2)

def ankle_center_y(kps):
    if kps is None: return None
    la, ra = kps.get("left_ankle"), kps.get("right_ankle")
    if la is None or ra is None: return None
    return float((la["y"] + ra["y"]) / 2)

def estimate_trochanter_height(h, ratio=0.53):
    if h is None or h <= 0: return None
    return float(h * ratio)
