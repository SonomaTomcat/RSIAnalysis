"""YOLO pose - SRP: only pose, drawing is separate concern but kept for convenience."""
import cv2, numpy as np
from core.interfaces import IPoseEstimator, Keypoints, BBox

COCO={5:"left_shoulder",6:"right_shoulder",7:"left_elbow",8:"right_elbow",
       9:"left_wrist",10:"right_wrist",11:"left_hip",12:"right_hip",
       13:"left_knee",14:"right_knee",15:"left_ankle",16:"right_ankle"}

class YOLOPoseEstimator(IPoseEstimator):
    def __init__(self, cfg):
        self.cfg=cfg; self.model=None; self.device=getattr(cfg, "device", "auto")
        self._load()
    def _load(self):
        try:
            from ultralytics import YOLO
            self.model=YOLO(self.cfg.model_path)
            dev=self.device
            to_dev = "cuda:0" if dev == "0" else dev
            if dev in ("mps","cuda","cuda:0","mps:0","0","xpu"):
                try: self.model.to(to_dev)
                except: pass
        except Exception as e:
            print(f"[Pose] load fail {e}")
    def estimate(self, roi) -> Keypoints|None:
        if roi is None or roi.size==0 or self.model is None: return None
        h,w=roi.shape[:2]
        try:
            predict_kwargs=dict(conf=self.cfg.conf, verbose=False)
            if self.device and self.device not in ("auto","cpu"):
                predict_kwargs["device"]=self.device
            res=self.model.predict(roi, **predict_kwargs)
            if not res or len(res[0].keypoints)==0 or res[0].keypoints.xy is None: return None
            boxes=res[0].boxes
            kpts=res[0].keypoints
            if boxes is None or len(boxes)==0: return None
            confs=boxes.conf.cpu().numpy()
            idx=int(np.argmax(confs))
            if hasattr(kpts,'xyn') and kpts.xyn is not None and len(kpts.xyn)>idx:
                xy=kpts.xyn[idx].cpu().numpy(); kc=kpts.conf[idx].cpu().numpy() if hasattr(kpts,'conf') else np.ones(17)
            else:
                xy=kpts.xy[idx].cpu().numpy()/np.array([w,h]); kc=np.ones(17)
            data={}
            for c,name in COCO.items():
                if kc[c]<0.2 and ("ankle" in name and kc[c]<0.05): data[name]=None; continue
                x,y=float(xy[c][0]),float(xy[c][1])
                if not 0<=x<=1 or not 0<=y<=1: data[name]=None; continue
                data[name]={"x":x,"y":y,"visibility":float(kc[c]),"px":int(x*w),"py":int(y*h)}
            # Hip is core, upper body can still be drawn if ankle missing (interpolated later for RSI)
            if data.get("left_hip") is None or data.get("right_hip") is None: return None
            return Keypoints(data)
        except Exception as e:
            print(f"[Pose] {e}"); return None
    def draw(self, image, kps:Keypoints, bbox:BBox):
        if kps is None or bbox is None: return image
        try:
            # Points: limbs only, no face
            for n in ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                      "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
                p=kps.get(n)
                if p is None: continue
                px=int(bbox.x1+p["x"]*bbox.w); py=int(bbox.y1+p["y"]*bbox.h)
                if "shoulder" in n: col=(255,0,255)
                elif "elbow" in n: col=(255,128,0)
                elif "wrist" in n: col=(0,128,255)
                elif "hip" in n: col=(0,255,255)
                elif "knee" in n: col=(0,255,0) if "left" in n else (255,180,0)
                else: col=(0,255,0) if "left" in n else (255,180,0)
                cv2.circle(image,(px,py),4,col,-1); cv2.circle(image,(px,py),6,(255,255,255),1)
            # Lines: limb skeleton only, no face
            for a,b in [("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
                        ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
                        ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
                        ("left_hip","right_hip"),("left_hip","left_knee"),("left_knee","left_ankle"),
                        ("right_hip","right_knee"),("right_knee","right_ankle")]:
                pa, pb=kps.get(a), kps.get(b)
                if pa and pb:
                    xa=int(bbox.x1+pa["x"]*bbox.w); ya=int(bbox.y1+pa["y"]*bbox.h)
                    xb=int(bbox.x1+pb["x"]*bbox.w); yb=int(bbox.y1+pb["y"]*bbox.h)
                    cv2.line(image,(xa,ya),(xb,yb),(0,255,255),2)
            return image
        except: return image
