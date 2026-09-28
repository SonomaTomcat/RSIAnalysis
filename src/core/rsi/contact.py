"""SRP: contact detection (ankle collapse + shoe deform) only."""
import numpy as np

def interpolate_nans(arr):
    arr=np.array(arr,dtype=float)
    nans=np.isnan(arr)
    if np.all(nans): return arr
    x=np.arange(len(arr))
    arr[nans]=np.interp(x[nans], x[~nans], arr[~nans])
    return arr

class AnkleCollapseContactDetector:
    """Inheritance-ready, decoupled. Detects ground vs collapse."""
    def __init__(self, vel_thresh=0.34, ground_percentile=68):  # 0.38 for longer contact
        self.vel_thresh=vel_thresh; self.ground_percentile=ground_percentile
    def detect(self, ankle_signal, fs):
        if ankle_signal is None or len(ankle_signal)<5: return None,None
        arr=np.array([float(v) if v is not None and not np.isnan(v) else np.nan for v in ankle_signal],dtype=float)
        if np.all(np.isnan(arr)): return None,None
        arr=interpolate_nans(arr)
        thr=np.percentile(arr, self.ground_percentile)
        vel=np.gradient(arr)*fs
        ground=arr >= (thr-0.008)
        collapse=(arr >= (thr-0.008)) & (np.abs(vel) < self.vel_thresh)
        try:
            from scipy.ndimage import binary_closing
            ground=binary_closing(ground, structure=np.ones(max(1,int(fs*0.04))))
            collapse=binary_closing(collapse, structure=np.ones(max(1,int(fs*0.04))))
        except: pass
        return ground, collapse
