"""SRP: RSI calculation only. Depends on abstractions, not concretions."""
import numpy as np
from core.signal.filter import ButterworthFilter
from core.signal.peak import PeakDetector
from core.rsi.contact import AnkleCollapseContactDetector
from dataclasses import dataclass

G=9.81

def _interp(arr):
    arr=np.array(arr,dtype=float)
    nans=np.isnan(arr)
    if np.all(nans): return arr
    x=np.arange(len(arr))
    arr[nans]=np.interp(x[nans], x[~nans], arr[~nans])
    return arr

class RSICalculator:
    """Clean, testable, single responsibility."""
    def __init__(self, cfg, filter: ButterworthFilter=None, peak_detector: PeakDetector=None, contact_detector: AnkleCollapseContactDetector=None):
        self.cfg=cfg
        self.filter=filter or ButterworthFilter(cfg.lowcut, cfg.highcut, use_lowpass=cfg.use_lowpass)
        self.peak=peak_detector or PeakDetector(cfg.prominence, cfg.distance_sec)
        self.contact=contact_detector or AnkleCollapseContactDetector()
    def _estimate_trochanter(self, h, t):
        if t and t>0: return float(t)
        if h and h>0: return float(h*0.53)
        return None
    def calculate(self, hip_signal, fs, ankle_signal=None, user_height_cm=None, trochanter_cm=None):
        if hip_signal is None or len(hip_signal)<10:
            return {"jump_count":0,"jumps":[],"mean_rsi":0,"best_rsi":0,"peaks":np.array([]),"troughs":np.array([]),"filtered":np.array([]),"raw":np.array([]),"fs":fs,"message":"Too short"}
        raw=np.array([float(v) if v is not None and not np.isnan(v) else np.nan for v in hip_signal],dtype=float)
        if np.all(np.isnan(raw)):
            return {"jump_count":0,"jumps":[],"mean_rsi":0,"best_rsi":0,"peaks":np.array([]),"troughs":np.array([]),"filtered":raw,"raw":raw,"fs":fs,"message":"No valid pose"}
        raw=_interp(raw)
        sig=1.0-raw - np.mean(1.0-raw)
        filtered=self.filter.apply(sig, fs)  # already zero-mean inside? ensure
        # ensure zero-mean
        filtered=filtered - np.mean(filtered)
        # we override filter's mean handling: use our filtered which is zero-mean
        # Actually ButterworthFilter returns zero-mean already
        peaks=self.peak.detect(filtered, fs)
        troughs=self.peak.detect(-filtered, fs)
        if len(peaks)==0 or len(troughs)<2:
            return {"jump_count":0,"jumps":[],"mean_rsi":0,"best_rsi":0,"peaks":peaks,"troughs":troughs,"filtered":filtered,"raw":raw,"fs":fs,"message":f"Insufficient peaks {len(peaks)} troughs {len(troughs)}"}
        peaks, troughs=np.sort(peaks), np.sort(troughs)
        while len(peaks) and len(troughs) and peaks[0]<troughs[0]: peaks=peaks[1:]
        while len(peaks) and len(troughs) and peaks[-1]>troughs[-1]: peaks=peaks[:-1]
        if len(peaks)==0 or len(troughs)<2:
            return {"jump_count":0,"jumps":[],"mean_rsi":0,"best_rsi":0,"peaks":peaks,"troughs":troughs,"filtered":filtered,"raw":raw,"fs":fs,"message":"Peak-trough mismatch"}
        # contact masks
        ground_mask=collapse_mask=None
        if ankle_signal is not None:
            try: ground_mask, collapse_mask=self.contact.detect(ankle_signal, fs)
            except: ground_mask=collapse_mask=None
        trochanter=self._estimate_trochanter(user_height_cm, trochanter_cm)
        jumps=[]
        for i in range(len(troughs)-1):
            t0,t1=troughs[i],troughs[i+1]
            cands=peaks[(peaks>t0)&(peaks<t1)]
            if len(cands)==0: continue
            pk=cands[int(np.argmax(filtered[cands]))] if len(cands)>1 else cands[0]
            v0,v1,vp=filtered[t0],filtered[t1],filtered[pk]
            thresh=(vp+(v0+v1)/2)/2
            # hip fallback
            takeoff_hip=next((k+1 for k in range(t0,pk) if filtered[k]<thresh<=filtered[k+1]), int((t0+pk)/2))
            landing_hip=next((k+1 for k in range(pk,t1) if filtered[k]>=thresh>filtered[k+1]), int((pk+t1)/2))
            t_flight_hip=(landing_hip-takeoff_hip)/fs if landing_hip>takeoff_hip else 0
            cycle=(t1-t0)/fs; t_contact_hip=cycle-t_flight_hip
            t_flight, t_contact, takeoff, landing = t_flight_hip, t_contact_hip, takeoff_hip, landing_hip
            # ankle/shoe refined: collapse_start -> ground_end, flight = ground_end+1 -> next ground_start
            if collapse_mask is not None and ground_mask is not None and len(collapse_mask)==len(filtered):
                try:
                    from scipy.ndimage import label
                    lc,_=label(collapse_mask); lg,_=label(ground_mask)
                    seg_c_t0=lc[t0] if 0<=t0<len(lc) else 0
                    seg_g_t0=lg[t0] if 0<=t0<len(lg) else 0
                    seg_g_t1=lg[t1] if 0<=t1<len(lg) else 0
                    collapse_start=None; takeoff_ankle=None; landing_ankle=None
                    if seg_c_t0!=0:
                        idx=np.where(lc==seg_c_t0)[0]; collapse_start=int(idx[0])
                        if seg_g_t0!=0:
                            idxg=np.where(lg==seg_g_t0)[0]; takeoff_ankle=int(idxg[-1]+1)
                    if collapse_start is None:
                        nxt=np.where(collapse_mask[t0:pk])[0]
                        if len(nxt)>0:
                            collapse_start=int(t0+nxt[0])
                            seg_g=lg[collapse_start] if 0<=collapse_start<len(lg) else 0
                            if seg_g!=0:
                                idxg=np.where(lg==seg_g)[0]; takeoff_ankle=int(idxg[-1]+1)
                    if seg_g_t1!=0:
                        idxg1=np.where(lg==seg_g_t1)[0]; landing_ankle=int(idxg1[0])
                    else:
                        nxt2=np.where(ground_mask[pk:t1])[0]
                        if len(nxt2)>0: landing_ankle=int(pk+nxt2[0])
                    if collapse_start is not None and takeoff_ankle is not None and landing_ankle is not None and landing_ankle>takeoff_ankle:
                        tc=(takeoff_ankle-collapse_start)/fs
                        tf=(landing_ankle-takeoff_ankle)/fs
                        if 0.06<=tc<=0.6 and 0.08<=tf<=1.0:
                            t_contact, t_flight, takeoff, landing = float(tc), float(tf), int(takeoff_ankle), int(landing_ankle)
                except: pass
            if not (0.08<=t_flight<=1.0 and 0.08<=t_contact<=1.0): continue
            rsi=t_flight/t_contact if t_contact else 0  # RSI = flight/contact (continuous CMJ)
            height=G*t_flight**2/8  # meters, keep 3 decimals in display
            if height<0.02: continue
            # TTT for literature RSImod: trough_before -> takeoff (not earlier unweighting)
            t_ttt=(takeoff - t0)/fs if takeoff>t0 else 0
            rsi_mod = height / t_ttt if t_ttt else 0  # RSImod = height(m) / TTT (literature)
            rsi_mod_contact = height / t_contact if t_contact else 0  # alternative: height/contact (dual column)
            jumps.append({"id":len(jumps)+1,"peak_idx":int(pk),"trough_before":int(t0),"trough_after":int(t1),"takeoff_idx":int(takeoff),"landing_idx":int(landing),"t_flight":float(t_flight),"t_contact":float(t_contact),"t_ttt":float(t_ttt),"rsi":float(rsi),"rsi_mod":float(rsi_mod),"rsi_mod_contact":float(rsi_mod_contact),"height":float(height),"height_trochanter_pred":float(height) if trochanter else None,"peak_val":float(vp),"thresh":float(thresh)})
        if not jumps:
            return {"jump_count":0,"jumps":[],"mean_rsi":0,"best_rsi":0,"mean_rsi_mod":0,"best_rsi_mod":0,"mean_ttt":0,"peaks":peaks,"troughs":troughs,"filtered":filtered,"raw":raw,"fs":fs,"message":"No valid jump segmented"}
        return {"jump_count":len(jumps),"jumps":jumps,"mean_rsi":float(np.mean([j["rsi"] for j in jumps])),"best_rsi":float(np.max([j["rsi"] for j in jumps])),"mean_rsi_mod":float(np.mean([j["rsi_mod"] for j in jumps])),"best_rsi_mod":float(np.max([j["rsi_mod"] for j in jumps])),"mean_ttt":float(np.mean([j["t_ttt"] for j in jumps])),"mean_contact":float(np.mean([j["t_contact"] for j in jumps])),"mean_flight":float(np.mean([j["t_flight"] for j in jumps])),"mean_height":float(np.mean([j["height"] for j in jumps])),"fatigue":float((np.mean([j["rsi"] for j in jumps[-3:]])-np.mean([j["rsi"] for j in jumps[:3]]))/np.mean([j["rsi"] for j in jumps[:3]])*100) if len(jumps)>=6 else 0.0,"peaks":peaks,"troughs":troughs,"filtered":filtered,"raw":raw,"fs":fs,"trochanter_cm":trochanter,"user_height_cm":user_height_cm,"height_note":"Pixel prediction with manual height/trochanter, prediction only, not accurate" if trochanter else "Physics prediction based on flight time, prediction only","height_warning":"Prediction only, not accurate, affected by camera angle/frame rate/calibration","contact_method":"ankle_collapse+shoe_deform" if collapse_mask is not None else "hip_threshold","message":"OK"}
