"""SRP: video IO only. Decoupled from detection."""
import cv2, tempfile, os

class VideoReader:
    def __init__(self, path): self.cap=cv2.VideoCapture(path)
    def __enter__(self): return self
    def __exit__(self,*_): self.cap.release()
    @property
    def fps(self):
        v=self.cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v>0 and not v!=v else 30.0
    @property
    def count(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    @property
    def wh(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def frames(self):
        idx=0
        while True:
            ok,frame=self.cap.read()
            if not ok: break
            yield idx, frame
            idx+=1

class VideoWriter:
    def __init__(self, path, fps, wh):
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        self.w=cv2.VideoWriter(path, fourcc, fps, wh)
        if not self.w.isOpened():
            fourcc=cv2.VideoWriter_fourcc(*'avc1')
            self.w=cv2.VideoWriter(path, fourcc, fps, wh)
    def write(self, frame): self.w.write(frame)
    def release(self): self.w.release()
