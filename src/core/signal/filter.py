"""SRP: filtering only."""
import numpy as np
try: from scipy.signal import butter, filtfilt; HAS=True
except: HAS=False

class ButterworthFilter:
    def __init__(self, lowcut=0.3, highcut=8.0, order=2, use_lowpass=True):
        self.lowcut=lowcut; self.highcut=highcut; self.order=order; self.use_lowpass=use_lowpass
    def apply(self, data: np.ndarray, fs: float) -> np.ndarray:
        if not HAS or len(data)<10: return data - np.mean(data)
        nyq=0.5*fs
        hc=min(self.highcut, nyq*0.95); lc=min(self.lowcut, nyq*0.4)
        if hc<=0 or lc>=hc: return data - np.mean(data)
        try:
            if self.use_lowpass:
                b,a=butter(self.order, hc/nyq, btype='low')
            else:
                b,a=butter(self.order, [lc/nyq, hc/nyq], btype='band')
            if len(data) <= 3*max(len(a),len(b)): return data - np.mean(data)
            f=filtfilt(b,a,data)
            return f - np.mean(f)
        except: return data - np.mean(data)
    # simplified: we want zero-mean filtered
    def apply_zero_mean(self, data, fs):
        f=self.apply(data, fs)
        return f - np.mean(f)
