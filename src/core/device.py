"""Device auto-detection for Apple MPS, NVIDIA CUDA, AMD ROCm, Intel/AMD integrated."""
import functools
import platform as plat

@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """
    Returns device string for ultralytics YOLO:
    - "mps"  : Apple Silicon M-chip (Metal)
    - "cuda" : NVIDIA CUDA or AMD ROCm (both via torch.cuda)
    - "xpu"  : Intel discrete/integrated (XPU)
    - "cpu"  : Fallback (Intel/AMD integrated without XPU, or no GPU)
    Priority: MPS > CUDA/ROCm > XPU > CPU
    """
    try:
        import torch
        # 1. Apple MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # 2. CUDA / ROCm (NVIDIA & AMD discrete via torch.cuda)
        #    AMD ROCm also exposes via torch.cuda and torch.version.hip
        if torch.cuda.is_available():
            # Distinguish AMD vs NVIDIA for info, but device string is still "cuda" for ultralytics
            return "cuda"
        # 3. AMD ROCm explicit check (hip)
        if hasattr(torch.version, "hip") and getattr(torch.version, "hip", None) is not None:
            # ROCm build without cuda.is_available() (rare)
            return "cuda"
        # 4. Intel XPU (Arc / integrated UHD/Iris Xe)
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
            return "xpu"
        # 5. OpenVINO for Intel integrated (fallback check)
        try:
            import openvino
            # If OpenVINO is installed, Intel iGPU can be used via OpenVINO, but torch will still be cpu
            # We keep device as cpu but label will mention OpenVINO
            pass
        except: pass
        # 6. DirectML for Windows integrated (AMD/Intel)
        try:
            import torch_directml  # noqa
            if torch_directml.is_available():
                return "privateuseone"  # ultralytics may not support, fallback to cpu
        except: pass
    except Exception:
        pass
    return "cpu"

def _is_hip() -> bool:
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except: return False

def get_device_info() -> dict:
    """Detailed info for UI display - covers CUDA/AMD/Intel."""
    dev = get_device()
    info = {"device": dev, "platform": plat.machine()}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["hip"] = getattr(torch.version, "hip", None)
        # Detect vendor for CUDA
        if dev == "cuda":
            try:
                name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                # Distinguish AMD vs NVIDIA
                if _is_hip() or "AMD" in name or "Radeon" in name:
                    info["label"] = f"AMD ROCm ({name})"
                    info["vendor"] = "AMD"
                elif "NVIDIA" in name or "GeForce" in name or "RTX" in name or "GTX" in name:
                    info["label"] = f"NVIDIA CUDA ({name})"
                    info["vendor"] = "NVIDIA"
                else:
                    info["label"] = f"CUDA ({name})"
                    info["vendor"] = "CUDA"
            except Exception:
                info["label"] = "CUDA/ROCm GPU"
                info["vendor"] = "CUDA"
        elif dev == "mps":
            info["label"] = "Apple MPS (M-chip GPU)"
            info["vendor"] = "Apple"
        elif dev == "xpu":
            try:
                # torch.xpu.get_device_name not always available
                info["label"] = "Intel XPU (Arc / Integrated)"
            except:
                info["label"] = "Intel XPU"
            info["vendor"] = "Intel"
        else:  # cpu
            # Check for integrated GPU presence for info only (even though running on CPU)
            integrated = []
            try:
                import cpuinfo  # noqa
            except: pass
            # Heuristic: check platform and try to detect iGPU
            # On Windows/Linux, integrated Intel UHD / AMD Radeon Graphics often present
            # We just label CPU with hint
            # Try to detect via lspci / system_profiler (best effort, no fail)
            try:
                import subprocess, sys
                if sys.platform == "darwin":
                    # macOS without MPS -> Intel Mac
                    info["label"] = "CPU (Intel Mac, no MPS)"
                    info["vendor"] = "Intel"
                elif sys.platform.startswith("linux"):
                    # Try lspci
                    out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
                    if "Intel" in out and ("UHD" in out or "Iris" in out or "Graphics" in out):
                        integrated.append("Intel Integrated")
                    if "AMD" in out and ("Radeon" in out or "Graphics" in out):
                        integrated.append("AMD Integrated")
                    if integrated:
                        info["label"] = f"CPU (fallback, iGPU detected: {', '.join(integrated)} - use OpenVINO for accel)"
                        info["vendor"] = "Integrated"
                    else:
                        info["label"] = "CPU"
                        info["vendor"] = "CPU"
                elif sys.platform == "win32":
                    info["label"] = "CPU (DirectML/OpenVINO available for Intel/AMD iGPU)"
                    info["vendor"] = "Integrated"
                else:
                    info["label"] = "CPU"
            except:
                info["label"] = "CPU"
        info["torch_hip"] = str(info.get("hip")) if info.get("hip") else None
    except Exception as e:
        info["label"] = dev
        info["error"] = str(e)
    return info

def get_optimal_device_for_yolo() -> str:
    """
    Returns device string directly usable for ultralytics YOLO predict(device=...).
    Normalizes to ultralytics expected values: "mps", "cpu", 0, "cuda:0" etc.
    For this project we return "mps"/"cuda"/"cpu"/"xpu" as ultralytics handles all.
    """
    d = get_device()
    # Ultralytics YOLO: device can be int, str, list. For simplicity return str.
    # For CUDA, ultralytics prefers 0 or "0" or "cuda:0", but "cuda" also works (maps to 0)
    if d == "cuda":
        return "0"  # most compatible: 0 means cuda:0
    return d
