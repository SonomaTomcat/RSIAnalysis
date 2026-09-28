"""YOLO detector - SRP: only detection, tracking delegated to Smoother."""
import numpy as np
from core.interfaces import IDetector, BBox

class BBoxSmoother:
    """SRP: EMA smoothing only."""
    def __init__(self, alpha=0.6): self.alpha=alpha; self.prev=None
    def smooth(self, cur: BBox) -> BBox:
        if self.prev and cur.iou(self.prev) > 0.1:
            a=self.alpha
            cur=BBox(int(a*cur.x1+(1-a)*self.prev.x1), int(a*cur.y1+(1-a)*self.prev.y1),
                     int(a*cur.x2+(1-a)*self.prev.x2), int(a*cur.y2+(1-a)*self.prev.y2))
        self.prev=cur
        return cur
    def reset(self): self.prev=None

class YOLODetector(IDetector):
    """Inheritance: IDetector <- YOLODetector. Decoupled via Config. Auto GPU (MPS/CUDA)."""
    def __init__(self, cfg):
        self.cfg=cfg; self.model=None; self.smoother=BBoxSmoother(cfg.smooth_alpha)
        self.device=getattr(cfg, "device", "auto")
        self._load()
    def _load(self):
        try:
            from ultralytics import YOLO
            self.model=YOLO(self.cfg.model_path)
            # Move model to device if supported (ultralytics handles MPS/CUDA/ROCm/XPU)
            dev=self.device
            # Normalize "0" -> "cuda:0" for .to()
            to_dev = "cuda:0" if dev == "0" else dev
            if dev in ("mps","cuda","cuda:0","mps:0","0","xpu"):
                try: self.model.to(to_dev)
                except: pass
        except Exception as e:
            print(f"[YOLO] load fail {e}"); self.model=None
    def detect(self, frame) -> BBox|None:
        h,w=frame.shape[:2]
        if self.model is None:
            return BBox(int(w*0.1),int(h*0.1),int(w*0.9),int(h*0.9))
        try:
            predict_kwargs=dict(conf=self.cfg.conf, iou=self.cfg.iou, classes=[0], verbose=False, max_det=5)
            if self.device and self.device not in ("auto","cpu"):
                predict_kwargs["device"]=self.device  # "mps", "0" (cuda), "xpu" all supported by ultralytics
            res=self.model.predict(frame, **predict_kwargs)
            if not res or len(res[0].boxes)==0:
                return self.smoother.prev
            boxes=res[0].boxes
            cx0,cy0=w/2,h/2
            best,score=None,-1
            for i,box in enumerate(boxes):
                x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                conf=float(box.conf[0].cpu().numpy())
                area=(x2-x1)*(y2-y1)
                cx,cy=(x1+x2)/2,(y1+y2)/2
                dist=np.sqrt((cx-cx0)**2+(cy-cy0)**2)/np.sqrt(w*w+h*h)
                s=area*conf*(1-0.3*dist)
                if s>score: score=s; best=BBox(int(x1),int(y1),int(x2),int(y2))
            if best is None:
                return self.smoother.prev
            best=best.pad().clip(w,h)
            return self.smoother.smooth(best)
        except Exception as e:
            print(f"[YOLO] infer {e}"); return self.smoother.prev
    def reset(self): self.smoother.reset()
