"""Abstract interfaces - Dependency Inversion, SRP."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# ---- Value Objects ----
@dataclass(frozen=True)
class BBox:
    x1: int; y1: int; x2: int; y2: int
    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1
    def pad(self, ratio=0.15, min_size=100) -> "BBox":
        pad_x, pad_y = int(self.w*ratio), int(self.h*ratio)
        if self.w < min_size or self.h < min_size:
            extra = (min_size - min(self.w, self.h)) // 2
            pad_x, pad_y = max(pad_x, extra), max(pad_y, extra)
        return BBox(self.x1-pad_x, self.y1-pad_y, self.x2+pad_x, self.y2+pad_y)
    def clip(self, W, H) -> "BBox":
        return BBox(max(0,self.x1), max(0,self.y1), min(W,self.x2), min(H,self.y2))
    def iou(self, other: "BBox") -> float:
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0,x2-x1)*max(0,y2-y1)
        return inter / (self.w*self.h + other.w*other.h - inter + 1e-6)

@dataclass
class Keypoints:
    """ROI-normalized 0-1, single person."""
    data: dict  # name -> {x,y,visibility}
    def get(self, name): return self.data.get(name)
    def has(self, *names): return all(self.data.get(n) is not None for n in names)

@dataclass
class Jump:
    id: int; takeoff_idx: int; landing_idx: int; peak_idx: int
    t_flight: float; t_contact: float; rsi: float; height: float

# ---- Interfaces ----
class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[BBox]: ...
    @abstractmethod
    def reset(self): ...

class IPoseEstimator(ABC):
    @abstractmethod
    def estimate(self, roi: np.ndarray) -> Optional[Keypoints]: ...
    def draw(self, image: np.ndarray, kps: Keypoints, bbox: BBox) -> np.ndarray: return image

class IFilter(ABC):
    @abstractmethod
    def apply(self, data: np.ndarray, fs: float) -> np.ndarray: ...

class IPeakDetector(ABC):
    @abstractmethod
    def detect(self, signal: np.ndarray, fs: float) -> np.ndarray: ...

class IContactDetector(ABC):
    @abstractmethod
    def detect(self, ankle_signal: list, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """return (ground_mask, collapse_mask)"""
        ...

class IRSICalculator(ABC):
    @abstractmethod
    def calculate(self, hip_signal: list, fs: float, ankle_signal=None) -> dict: ...
