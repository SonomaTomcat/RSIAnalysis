"""Simple Factory + DI - decouples creation."""
from core.config import DetectorConfig, PoseConfig, RSIConfig
from core.detection.yolo_detector import YOLODetector
from core.pose.yolo_pose import YOLOPoseEstimator
from core.signal.filter import ButterworthFilter
from core.signal.peak import PeakDetector
from core.rsi.contact import AnkleCollapseContactDetector
from core.rsi.calculator import RSICalculator
from core.device import get_device

def _resolve_device(cfg_device: str) -> str:
    if cfg_device and cfg_device != "auto":
        return cfg_device
    # Use optimal YOLO device (0 for cuda, mps for Apple, etc.)
    from core.device import get_optimal_device_for_yolo
    return get_optimal_device_for_yolo()

def create_detector(cfg=DetectorConfig()):
    cfg.device = _resolve_device(cfg.device)
    return YOLODetector(cfg)
def create_pose(cfg=PoseConfig()):
    cfg.device = _resolve_device(cfg.device)
    return YOLOPoseEstimator(cfg)
def create_rsi_calculator(cfg=RSIConfig()):
    return RSICalculator(cfg,
        filter=ButterworthFilter(cfg.lowcut, cfg.highcut, use_lowpass=cfg.use_lowpass),
        peak_detector=PeakDetector(cfg.prominence, cfg.distance_sec),
        contact_detector=AnkleCollapseContactDetector())
