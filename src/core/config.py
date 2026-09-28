"""Single source of config - SRP."""
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf: float = 0.30
    iou: float = 0.45
    smooth_alpha: float = 0.6
    device: str = "auto"  # auto | cpu | mps | cuda | cuda:0

@dataclass
class PoseConfig:
    model_path: str = "yolov8n-pose.pt"
    conf: float = 0.30
    device: str = "auto"

@dataclass
class RSIConfig:
    lowcut: float = 0.3
    highcut: float = 8.0
    prominence: float = 0.5
    distance_sec: float = 0.35
    use_lowpass: bool = True

@dataclass
class AppConfig:
    user_height_cm: int = 180
    trochanter_cm: float = 95.4  # manual input, default 180*0.53
    show_bbox: bool = True
    show_skeleton: bool = True
