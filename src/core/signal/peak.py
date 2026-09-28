"""SRP: peak detection only."""
import numpy as np
try: from scipy.signal import find_peaks; HAS=True
except: HAS=False

class PeakDetector:
    def __init__(self, prominence_factor=0.5, distance_sec=0.35):
        self.prominence=prominence_factor; self.distance_sec=distance_sec
    def detect(self, signal: np.ndarray, fs: float) -> np.ndarray:
        if len(signal)<5: return np.array([],dtype=int)
        if HAS:
            try:
                prom=max(np.std(signal)*self.prominence,0.02)
                dist=int(fs*self.distance_sec)
                peaks,_=find_peaks(signal, distance=dist, prominence=prom, width=1)
                return peaks
            except: pass
        peaks=[]
        for i in range(1,len(signal)-1):
            if signal[i]>signal[i-1] and signal[i]>signal[i+1] and signal[i]>np.mean(signal):
                if not peaks or i-peaks[-1]>=int(fs*self.distance_sec):
                    peaks.append(i)
        return np.array(peaks,dtype=int)
